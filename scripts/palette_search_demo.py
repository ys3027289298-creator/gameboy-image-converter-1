"""Branch-and-bound 4-color palette search demo on four small synthetic assets.

Reproduces the "manual four-color tweaking is unstable -> exhaustive optimum"
workflow end to end:

  * every asset is generated in-process (no files, no network, tiny 16x16);
  * the candidate set and pruning tolerance are configurable per run;
  * each asset is scored both with the full (unpruned) enumeration and with
    lower-bound pruning, so pre/post-pruning node counts are directly
    comparable;
  * a deliberately hand-picked palette is scored on the identical mean-dE
    objective and diffed against the optimum;
  * the tolerance and time-limit knobs get a dedicated demonstration, the
    latter showing the explicit timeout return.

Run:  venv/bin/python scripts/palette_search_demo.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import gb_pipeline as gbp


def _hex(palette: np.ndarray) -> list:
    return ["#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b)) for r, g, b in palette]


def make_assets() -> dict:
    """Four small deterministic 16x16 RGB assets covering distinct content."""
    rng = np.random.RandomState(11)

    # 1. Flat logo: exactly four flat brand colors in quadrants.
    spots = np.array(
        [[232, 60, 60], [60, 120, 232], [240, 220, 90], [40, 40, 48]], dtype=np.uint8
    )
    yy, xx = np.mgrid[0:16, 0:16]
    region = (yy >= 8).astype(int) * 2 + (xx >= 8).astype(int)
    flat_logo = spots[region]

    # 2. Gray ramp: eight discrete gray levels, horizontal bands.
    levels = np.linspace(20, 235, 8, dtype=np.uint8)
    gray_ramp = np.repeat(levels[:, None, None], 16, axis=1)
    gray_ramp = np.repeat(gray_ramp, 2, axis=0)
    gray_ramp = np.broadcast_to(gray_ramp, (16, 16, 1)).astype(np.uint8)
    gray_ramp = np.repeat(gray_ramp, 3, axis=2).copy()

    # 3. Hue blocks: ten flat multi-hue colors in a 2-row x 5-column tiling.
    hue_colors = np.array(
        [
            [220, 40, 40], [230, 130, 30], [220, 210, 40], [40, 180, 70],
            [30, 200, 200], [40, 90, 220], [140, 50, 210], [230, 90, 180],
            [240, 240, 240], [30, 30, 30],
        ],
        dtype=np.uint8,
    )
    cols = np.minimum(np.arange(16) // 3, 4)          # 0..4 across the width
    rows = (np.arange(16) // 8) * 5                    # second row shifts by 5
    tile_idx = rows[:, None] + cols[None, :]
    hue_blocks = hue_colors[tile_idx]

    # 4. Noisy mix: smooth gradient plus seeded noise -> many soft colors.
    gx = np.linspace(0, 255, 16, dtype=np.float64)
    base = np.stack(
        [np.broadcast_to(gx, (16, 16)), np.broadcast_to(gx[::-1, None], (16, 16)),
         np.broadcast_to(128.0, (16, 16))],
        axis=-1,
    )
    noisy = np.clip(base + rng.normal(0, 22, (16, 16, 3)), 0, 255).astype(np.uint8)

    return {
        "flat_logo (4 flat brand colors)": flat_logo,
        "gray_ramp (8 gray levels)": gray_ramp,
        "hue_blocks (10 flat hues)": hue_blocks,
        "noisy_mix (gradient + noise)": noisy,
    }


def candidate_set(image: np.ndarray, max_colors: int = 12) -> np.ndarray:
    """Configurable candidate pool, bounded so full enumeration stays tractable.

    The pool is the image's own working set (PIL MEDIANCUT then RGB555 snap,
    the pipeline's standard ``quantize_working_set`` step) capped at
    ``max_colors``; a free candidate pool of arbitrary size would make
    C(n, 4) enumeration meaningless, which is precisely why the candidate set
    is an explicit, caller-configurable input to the search.
    """
    from PIL import Image

    quantized = gbp.quantize_working_set(
        Image.fromarray(image, "RGB"), max_colors
    )
    return np.unique(np.asarray(quantized, dtype=np.uint8).reshape(-1, 3), axis=0)


def hand_palette(image: np.ndarray) -> np.ndarray:
    """A deliberately hand-picked (sub-optimal but plausible) 4-color palette."""
    uniq = np.unique(image.reshape(-1, 3), axis=0)
    # Evenly spaced pick across the source colors -- a typical manual tweak,
    # which ignores both pixel frequency and perceptual Lab spacing.
    idx = np.linspace(0, len(uniq) - 1, 4).round().astype(int)
    return gbp.snap_rgb555(uniq[idx])


def main() -> None:
    assets = make_assets()
    print("=" * 96)
    print("EXHAUSTIVE 4-COLOR PALETTE SEARCH (objective: pixel-weighted mean CIELAB dE)")
    print("=" * 96)

    summary = []
    for name, image in assets.items():
        cands = candidate_set(image)
        manual = hand_palette(image)
        manual_de = gbp.palette_mean_delta_e(image, manual)

        # Unpruned enumeration = the "before pruning" baseline.
        t0 = time.perf_counter()
        full = gbp.search_four_color_palette(image, cands, enable_pruning=False)
        t_full = time.perf_counter() - t0

        # Branch-and-bound with the provable lower bound = "after pruning".
        pruned = gbp.search_four_color_palette(
            image, cands, enable_pruning=True, warm_start=manual
        )
        assert abs(pruned.total_delta_e - full.total_delta_e) < 1e-9, (
            "pruning changed the optimum -- lower bound is unsound"
        )

        print(f"\n### {name}")
        print(f"  candidate colors : {pruned.candidate_count}  "
              f"(full enumeration would visit {full.nodes_total} nodes / "
              f"{full.leaves_evaluated} complete 4-color combinations)")
        print(f"  hand palette     : {_hex(gbp.luminance_sort(manual))}  mean dE = {manual_de:7.3f}")
        print(f"  optimal palette  : {_hex(pruned.palette)}  mean dE = {pruned.mean_delta_e:7.3f}")
        print(f"  dE improvement   : {manual_de - pruned.mean_delta_e:7.3f} "
              f"({100 * (manual_de - pruned.mean_delta_e) / max(manual_de, 1e-9):4.1f}% lower than hand)")
        print(f"  nodes before prune: {full.nodes_evaluated:6d}  "
              f"after prune: {pruned.nodes_evaluated:6d}  "
              f"subtrees pruned: {pruned.nodes_pruned:5d}  "
              f"({100 * pruned.nodes_pruned / max(full.nodes_evaluated, 1):4.1f}% of baseline nodes saved)")
        print(f"  timing           : unpruned {t_full * 1e3:6.2f} ms  "
              f"pruned {pruned.elapsed_seconds * 1e3:6.2f} ms  "
              f"status={pruned.status}")
        kept = {tuple(int(v) for v in c) for c in pruned.palette}
        hand_set = {tuple(int(v) for v in c) for c in gbp.luminance_sort(manual)}
        print(f"  palette diff     : {len(kept & hand_set)}/4 colors shared with the hand palette; "
              f"replaced {sorted('#%02X%02X%02X' % c for c in hand_set - kept)} -> "
              f"{sorted('#%02X%02X%02X' % c for c in kept - hand_set)}")
        summary.append((name, pruned.mean_delta_e, manual_de,
                        full.nodes_evaluated, pruned.nodes_evaluated, pruned.nodes_pruned,
                        pruned.elapsed_seconds))

    # --- configurable tolerance demonstration -------------------------------
    # A dedicated two-cluster dataset: gain-ordered DFS reaches the optimum
    # late, so a positive tolerance provably trades a small dE gap for fewer
    # evaluated nodes. The returned gap is guaranteed <= the tolerance.
    cluster_red = np.array(
        [[200, 30, 30], [210, 40, 40], [190, 25, 35], [205, 35, 25]], dtype=np.uint8
    )
    cluster_blue = np.array(
        [[30, 30, 200], [40, 40, 210], [25, 35, 190], [35, 25, 205]], dtype=np.uint8
    )
    middle = np.array(
        [[120, 120, 120], [130, 110, 125], [110, 125, 130], [125, 130, 110]],
        dtype=np.uint8,
    )
    tol_cands = gbp.snap_rgb555(np.vstack([cluster_red, cluster_blue, middle]))
    tol_cands = np.unique(tol_cands, axis=0)
    rng_tol = np.random.RandomState(5)
    tol_img = np.vstack(
        [cluster_red[rng_tol.randint(0, 4, (32, 3))],
         cluster_blue[rng_tol.randint(0, 4, (32, 3))]]
    )
    exact = gbp.search_four_color_palette(tol_img, tol_cands)
    print("\n" + "-" * 96)
    print(f"CONFIGURABLE PRUNING TOLERANCE ({len(tol_cands)} candidates, "
          "two far-apart color clusters)")
    print(f"  {'tolerance':>9s} {'mean dE':>9s} {'gap':>8s} {'nodes':>6s}  status")
    print(f"  {'(exact)':>9s} {exact.mean_delta_e:9.3f} {0.0:8.3f} "
          f"{exact.nodes_evaluated:6d}  {exact.status}")
    for tol in (20.0, 40.0):
        approx = gbp.search_four_color_palette(tol_img, tol_cands, tolerance=tol)
        gap = approx.mean_delta_e - exact.mean_delta_e
        assert gap <= tol + 1e-9  # the tolerance contract
        print(f"  {tol:9.1f} {approx.mean_delta_e:9.3f} {gap:8.3f} "
              f"{approx.nodes_evaluated:6d}  {approx.status}")

    # --- explicit timeout return demonstration ------------------------------
    rng = np.random.RandomState(3)
    big_cands = gbp.snap_rgb555(rng.randint(0, 256, (26, 3)).astype(np.uint8))
    big_cands = np.unique(big_cands, axis=0)
    big_img = big_cands[rng.randint(0, len(big_cands), (8, 8))]
    print("\n" + "-" * 96)
    print("SEARCH TIMEOUT (candidate set = %d colors -> %d full combinations)"
          % (len(big_cands), int(__import__("math").comb(len(big_cands), 4))))
    for limit in (None, 0.002):
        res = gbp.search_four_color_palette(
            big_img, big_cands, time_limit=limit, warm_start=big_cands[:4]
        )
        limit_txt = "no limit" if limit is None else f"{limit * 1e3:.1f} ms budget"
        palette_txt = "None" if res.palette is None else _hex(res.palette)
        print(f"  {limit_txt:>14s}: status={res.status:22s} timed_out={res.timed_out!s:5s} "
              f"dE={res.mean_delta_e:7.3f} nodes={res.nodes_evaluated:6d} "
              f"elapsed={res.elapsed_seconds * 1e3:6.2f} ms palette={palette_txt}")

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"{'asset':34s} {'opt dE':>8s} {'hand dE':>9s} {'nodes pre':>10s} "
          f"{'nodes post':>11s} {'pruned':>8s} {'time ms':>8s}")
    for row in summary:
        print(f"{row[0]:34s} {row[1]:8.3f} {row[2]:9.3f} {row[3]:10d} "
              f"{row[4]:11d} {row[5]:8d} {row[6] * 1e3:8.2f}")


if __name__ == "__main__":
    main()
