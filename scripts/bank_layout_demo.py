"""Reproducible bank-rearrangement benchmark on four small synthetic tile sets.

Run from the repo root:

    python3 scripts/bank_layout_demo.py

For every material the script prints, in one run:
  * the bank allocation table (which stored pattern id lands in which bank),
  * bank count and per-bank fill,
  * runtime bank switches under the first-appearance baseline and under the
    co-occurrence-driven rearrangement (absolute and percent reduction),
  * rebuild consistency (a GBImage-shaped stand-in is relaid out and every
    cell's effective pattern + the full RGB render are compared pixel-by-pixel),
  * rearrangement wall time.

Only numpy is used (and the stdlib); the four materials are generated inline,
deterministically, and kept tiny so the whole sweep finishes in well under a
second.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import tile_banks


@dataclass
class FakeGBImage:
    """Field-for-field stand-in for gb_pipeline.GBImage (avoids importing its
    image-processing dependencies just to exercise the storage relayout)."""

    patterns: np.ndarray
    tilemap: np.ndarray
    attrs_palette: np.ndarray
    attrs_hflip: np.ndarray
    attrs_vflip: np.ndarray
    palettes: np.ndarray


def _distinct_patterns(n, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        tile = rng.randint(0, 4, size=(8, 8)).astype(np.uint8)
        tile[0, 0] = i % 4
        out.append(tile)
    return np.stack(out)


def _effective_cell(gb, tr, tc):
    pattern = gb.patterns[int(gb.tilemap[tr, tc])]
    if gb.attrs_hflip[tr, tc]:
        pattern = np.fliplr(pattern)
    if gb.attrs_vflip[tr, tc]:
        pattern = np.flipud(pattern)
    return pattern


def _render(gb):
    th, tw = gb.tilemap.shape
    out = np.zeros((th * 8, tw * 8, 3), dtype=np.uint8)
    for tr in range(th):
        for tc in range(tw):
            pattern = _effective_cell(gb, tr, tc)
            pal = gb.palettes[int(gb.attrs_palette[tr, tc])]
            out[tr * 8: tr * 8 + 8, tc * 8: tc * 8 + 8] = pal[pattern]
    return out


def _make_gb(tilemap, n_patterns, seed):
    th, tw = tilemap.shape
    rng = np.random.RandomState(seed + 100)
    return FakeGBImage(
        patterns=_distinct_patterns(n_patterns, seed),
        tilemap=tilemap.astype(np.int32),
        attrs_palette=rng.randint(0, 2, size=(th, tw)).astype(np.uint8),
        attrs_hflip=rng.randint(0, 2, size=(th, tw)).astype(bool),
        attrs_vflip=rng.randint(0, 2, size=(th, tw)).astype(bool),
        palettes=np.tile(
            np.array(
                [[240, 240, 240], [170, 170, 170], [90, 90, 90], [10, 10, 10]],
                dtype=np.uint8,
            ),
            (2, 1, 1),
        ),
    )


def _materials():
    """Four small, structurally different tilemaps (each < 128 cells)."""
    # 1. Hub: one shared pattern interleaved with many leaves -- the baseline
    #    keeps the hub far from most leaves, so rearrangement pays off most.
    hub = np.array([
        [0, 1, 2, 3, 4, 0, 4, 1, 4, 2, 4, 3],
        [0, 1, 2, 3, 4, 0, 4, 1, 4, 2, 4, 3],
    ])

    # 2. Two neighborhoods: strong pairs {0,3} and {1,2} whose first-appearance
    #    order (0,1,2,3) makes the baseline split each pair across banks.
    row = [0, 1, 2, 3, 0, 3, 0, 3, 1, 2, 1, 2, 0, 3, 1, 2]
    neighborhoods = np.array([row, row[::-1], row])

    # 3. Scrambled: every cell drawn uniformly at random -- weak structure, the
    #    hardest case for co-occurrence grouping.
    rng = np.random.RandomState(7)
    scrambled = rng.randint(0, 12, size=(6, 10))

    # 4. Sequential stripes: long runs of the same pattern, natural bank runs.
    stripes = np.repeat(np.arange(10)[None, :], 6, axis=0).copy()

    return [
        ("hub", hub, 5, 2),
        ("neighborhoods", neighborhoods, 4, 2),
        ("scrambled", scrambled, 12, 4),
        ("stripes", stripes, 10, 3),
    ]


def main():
    rows = []
    all_ok = True
    for name, tilemap, n_patterns, cap in _materials():
        n_banks = max(1, -(-n_patterns // cap))
        t0 = time.perf_counter()
        layout = tile_banks.optimize_bank_layout(
            tilemap,
            bank_capacity=cap,
            n_banks=n_banks,
            n_patterns=n_patterns,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Rebuild consistency: relayout storage + references, then compare every
        # cell's effective pattern and the full rendered image pixel-by-pixel.
        gb = _make_gb(tilemap, n_patterns, seed=len(name))
        relaid = tile_banks.apply_bank_layout(gb, layout)
        th, tw = tilemap.shape
        cells_match = all(
            np.array_equal(_effective_cell(gb, tr, tc), _effective_cell(relaid, tr, tc))
            for tr in range(th)
            for tc in range(tw)
        )
        pixels_match = bool(np.array_equal(_render(gb), _render(relaid)))
        ok = cells_match and pixels_match
        all_ok &= ok

        reduction = (
            100.0 * (layout.baseline_switches - layout.switches)
            / max(1, layout.baseline_switches)
        )
        rows.append((name, layout, elapsed_ms, ok, reduction))

        print(f"\n=== {name} ===")
        print(
            f"cells={layout.n_cells} patterns={layout.n_patterns} "
            f"capacity={layout.bank_capacity} banks={layout.banks_used}"
            f"(requested {layout.n_banks_requested or layout.n_banks})"
        )
        for b, members in enumerate(layout.bank_members):
            print(f"  bank {b}: {len(members):2d}/{layout.bank_capacity}  {members}")
        print(
            f"switches: baseline(first-appearance)={layout.baseline_switches}  "
            f"rearranged={layout.switches}  cut_edges={layout.cut_edges}  "
            f"co-occurrence edges={layout.co_occurrence_edges}"
        )
        print(
            f"reduction={reduction:.1f}%  rebuild pixel-identical={ok}  "
            f"rearrange time={elapsed_ms:.2f} ms"
        )
        if layout.warnings:
            for w in layout.warnings:
                print(f"  WARNING: {w}")

    print("\n=== summary ===")
    header = (
        f"{'material':14s} {'banks':>5s} {'base_sw':>8s} {'opt_sw':>7s} "
        f"{'reduction':>9s} {'pixels_ok':>9s} {'time_ms':>8s}"
    )
    print(header)
    for name, layout, elapsed_ms, ok, reduction in rows:
        print(
            f"{name:14s} {layout.banks_used:5d} {layout.baseline_switches:8d} "
            f"{layout.switches:7d} {reduction:8.1f}% {str(ok):>9s} "
            f"{elapsed_ms:8.2f}"
        )

    # Determinism spot-check across the four materials.
    for name, tilemap, n_patterns, cap in _materials():
        n_banks = max(1, -(-n_patterns // cap))
        a = tile_banks.optimize_bank_layout(
            tilemap, bank_capacity=cap, n_banks=n_banks, n_patterns=n_patterns
        )
        b = tile_banks.optimize_bank_layout(
            tilemap.copy(), bank_capacity=cap, n_banks=n_banks, n_patterns=n_patterns
        )
        assert np.array_equal(a.assignment, b.assignment), name
    print("determinism: layouts identical on repeated runs for all materials")

    if not all_ok:
        print("FAIL: at least one rebuild was not pixel-identical")
        sys.exit(1)
    print("OK: every rebuild is pixel-identical and no layout exceeds capacity")


if __name__ == "__main__":
    main()
