"""Benchmark for the OAM per-scanline sprite budget.

Four small, fully synthetic frame sequences (no fixtures, no network, stdlib
+numpy only), each run both unrestricted and under the 10-sprites-per-scanline
limit. For every sequence it prints: frame drop rate, player-perceptible loss
counts, allocation/conversion time, and pixel difference versus the no-limit
composite. Run:

    venv/bin/python scripts/sprite_budget_bench.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import gb_pipeline as gbp
from gb_pipeline import ScanlineBudget, SpriteFrame, SpriteObject

W, H = 48, 32
PLAYER = (6.0, 16.0)
LIMIT = 10


def _sprite(sprite_id, x, y, w, h, color, rng):
    pixels = np.tile(np.array(color, dtype=np.uint8), (h, w, 1))
    # Sprites stay mostly opaque; a few random transparent cells model cutouts.
    transparent = np.zeros((h, w), dtype=bool)
    cut = rng.random((h, w)) < 0.05
    transparent |= cut
    return SpriteObject(x=x, y=y, pixels=pixels, transparent=transparent, id=sprite_id)


def _background(rng):
    base = rng.randint(0, 40, size=(H, W, 3)).astype(np.int16)
    return base.astype(np.uint8)


def _crowd_line(rng, n_frames, n_on_line, movers):
    """Sequence 1: a crowded scanline -- many 1xN sprites sharing one line."""
    frames = []
    for f in range(n_frames):
        sprites = []
        for i in range(n_on_line):
            x = (i * 3 + f) % (W - 2)
            sprites.append(_sprite(i, x, 8, 2, 6, (200, 60 + i * 10 % 150, 40), rng))
        for j in range(movers):
            sprites.append(_sprite(
                100 + j, (j * 7 + f * 2) % (W - 4), (f * 3 + j) % (H - 4),
                4, 4, (60, 120, 220), rng,
            ))
        frames.append(SpriteFrame(sprites=sprites, background=_background(rng)))
    return frames


def _roaming(rng, n_frames, n_sprites):
    """Sequence 2: a roaming swarm that crosses one horizontal band -- the
    sprites share only a few scanlines, so 18 of them overflow the 10/line
    cap even though the canvas is far from full."""
    frames = []
    band_top, band_h = 10, 6
    for f in range(n_frames):
        sprites = []
        for i in range(n_sprites):
            x = int((i * 5 + f * 3) % max(W - 4, 1))
            y = band_top + (i * 2 + f) % band_h
            sprites.append(_sprite(i % 8, x, y, 3, 3, (180, 90, 200), rng))
        frames.append(SpriteFrame(sprites=sprites, background=_background(rng)))
    return frames


def _mixed_sizes(rng, n_frames):
    """Sequence 3: a few big sprites amid many small ones (area weight matters);
    the small sprites are packed onto one scanline to force arbitration."""
    frames = []
    for f in range(n_frames):
        sprites = [
            _sprite(0, 2 + f, 4, 8, 8, (230, 200, 60), rng),
            _sprite(1, 30, 18 - f, 8, 8, (60, 220, 180), rng),
        ]
        for i in range(14):
            sprites.append(_sprite(
                10 + i, (i * 3 + f) % (W - 2), 0,
                2, 2, (220, 80, 120), rng,
            ))
        frames.append(SpriteFrame(sprites=sprites, background=_background(rng)))
    return frames


def _quiet(rng, n_frames):
    """Sequence 4: under-limit everywhere -- must show zero loss / zero diff."""
    frames = []
    for f in range(n_frames):
        sprites = [
            _sprite(i, (i * 9 + f) % (W - 4), (i * 5) % (H - 4),
                    4, 4, (90, 160, 90), rng)
            for i in range(6)
        ]
        frames.append(SpriteFrame(sprites=sprites, background=_background(rng)))
    return frames


def _run(name, frames, budget):
    t0 = time.perf_counter()
    res = gbp.allocate_sprite_budget(frames, PLAYER, budget)
    wall = time.perf_counter() - t0
    s = res.stats

    # Independent re-check of the reported diff against the unrestricted
    # reference, so the printed number is not trusted from the stats alone.
    re_diff = sum(
        int(np.count_nonzero(np.any(r != u, axis=2)))
        for r, u in zip(res.restricted, res.unrestricted)
    )
    assert re_diff == s["diff_pixels_total"], "diff metric self-check failed"

    print(
        f"{name:26s} frames={s['frames']:2d} sprites/frame~="
        f"{s['sprite_frame_instances'] / s['frames']:4.1f} "
        f"drop_rate={s['frame_drop_rate']:5.2f} "
        f"drop_frames={s['frames_with_drops']:2d} "
        f"perceptible={s['perceptible_losses']:3d} "
        f"lost_lines={s['lost_scanlines']:4d} "
        f"diff_px={s['diff_pixels_total']:5d} "
        f"mean_diff={s['mean_diff_fraction'] * 100:5.2f}% "
        f"max_diff={s['max_diff_fraction'] * 100:5.2f}% "
        f"time={max(wall, s['allocation_seconds']) * 1000:7.2f}ms"
    )


def main():
    rng = np.random.RandomState(20261118)
    sequences = [
        ("1 crowded scanline", _crowd_line(rng, 6, 14, 4)),
        ("2 roaming swarm", _roaming(rng, 8, 18)),
        ("3 mixed sizes", _mixed_sizes(rng, 6)),
        ("4 quiet (under limit)", _quiet(rng, 6)),
    ]
    print(f"canvas={W}x{H} player={PLAYER} limit={LIMIT} sprites/scanline\n")
    budget = ScanlineBudget(
        max_per_scanline=LIMIT,
        weight_distance=1.0,
        weight_frequency=1.0,
        weight_area=1.0,
    )
    for name, frames in sequences:
        _run(name, frames, budget)
    print("\nValidation check: illegal limits must raise ValueError ...")
    for bad in (0, -3, 1.5, True):
        try:
            ScanlineBudget(max_per_scanline=bad)
        except ValueError as exc:
            print(f"  ok  {bad!r:6} -> ValueError: {exc}")
        else:
            raise AssertionError(f"illegal limit {bad!r} was accepted")


if __name__ == "__main__":
    main()
