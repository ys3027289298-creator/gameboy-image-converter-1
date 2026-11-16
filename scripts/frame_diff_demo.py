"""Frame-difference codec demonstration on four small synthetic animations.

Run (no arguments, no network, stdlib + numpy only):

    venv/bin/python scripts/frame_diff_demo.py

Builds four 32x32 RGB animations (6 frames each) with deliberately different
motion profiles, encodes each as a baseline frame plus per-frame delta blocks
(gb_pipeline.encode_frame_diff, block_size=8, lossless change_threshold=0),
then reports for every animation:

  * delta-encoding payload volume vs. full per-frame storage;
  * per-frame changed-block counts and pixel-exact reconstruction consistency;
  * the MissingBaselineError raised when frame 0 is unavailable.

A fifth section shows the configurable change_threshold trading fidelity for
size. All frames are generated in-memory with a fixed numpy seed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import gb_pipeline as g

WIDTH = HEIGHT = 32          # 16 blocks of 8x8
N_FRAMES = 6
BLOCK = 8
BG = (12, 24, 48)


def _solid(color=BG):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :] = np.asarray(color, dtype=np.uint8)
    return frame


def make_blink():
    """Mostly static screen: a single 8x8 block blinks on/off every frame."""
    frames = [_solid()]
    for i in range(1, N_FRAMES):
        nxt = frames[-1].copy()
        nxt[0:BLOCK, 0:BLOCK] = (220, 200, 60) if i % 2 else BG
        frames.append(nxt)
    return frames


def make_sprite_walk():
    """One 8x8 sprite walks left->right along the top band; nothing else moves."""
    frames = []
    for i in range(N_FRAMES):
        frame = _solid()
        col = min(i * BLOCK, WIDTH - BLOCK)
        frame[0:BLOCK, col:col + BLOCK] = (220, 60, 60)
        frames.append(frame)
    return frames


def make_half_scroll():
    """Lower half of the screen scrolls every frame (8 of 16 blocks churn)."""
    frames = []
    for i in range(N_FRAMES):
        frame = _solid()
        for row in range(2, 4):  # block rows 2 and 3 -> lower half
            y0, y1 = row * BLOCK, (row + 1) * BLOCK
            shade = ((i * 40 + row * 50) % 256)
            frame[y0:y1, :] = (shade, shade // 2, 255 - shade)
        frames.append(frame)
    return frames


def make_full_churn():
    """Every block changes every frame: independent random pixel content."""
    rng = np.random.RandomState(20261117)
    return [rng.randint(0, 256, (HEIGHT, WIDTH, 3)).astype(np.uint8)
            for _ in range(N_FRAMES)]


def report(name, frames):
    seq = g.encode_frame_diff(frames, block_size=BLOCK, change_threshold=0)
    rebuilt = g.decode_frame_diff(seq)

    payload = g.frame_diff_payload_size(seq)
    full = g.frame_diff_full_size(seq)
    blob = g.serialize_frame_diff(seq)
    changed = g.frame_diff_changed_blocks(seq)
    mismatches = g.frame_diff_pixel_mismatch(frames, rebuilt)
    total_pixels = HEIGHT * WIDTH

    print(f"### {name}")
    print(f"  frames={len(frames)}  size={WIDTH}x{HEIGHT}  "
          f"blocks/frame={ (HEIGHT // BLOCK) * (WIDTH // BLOCK) }  "
          f"block_size={BLOCK}  change_threshold=0 (lossless)")
    print(f"  changed blocks per delta frame : {changed}")
    print(f"  differing pixels per frame     : {mismatches} "
          f"(of {total_pixels} pixels/frame)")
    print(f"  frame-diff payload             : {payload:6d} B")
    print(f"  serialized blob (with header)  : {len(blob):6d} B")
    print(f"  full per-frame storage         : {full:6d} B")
    print(f"  saving vs full storage         : {100.0 * (1 - payload / full):5.1f}%  "
          f"(ratio {payload / full:.3f})")
    assert mismatches == [0] * len(frames), "lossless encoding must reconstruct exactly"

    # Baseline loss: decode and deserialize must both refuse the stream.
    lost = g.FrameDiffSequence(
        baseline=None, deltas=seq.deltas, block_size=seq.block_size,
        change_threshold=seq.change_threshold, width=seq.width, height=seq.height,
    )
    for label, call in (
        ("decode_frame_diff", lambda: g.decode_frame_diff(lost)),
        ("deserialize_frame_diff",
         lambda: g.deserialize_frame_diff(g.serialize_frame_diff(lost))),
    ):
        try:
            call()
        except g.MissingBaselineError as exc:
            print(f"  baseline missing -> {label}: {type(exc).__name__}: "
                  f"{str(exc).split(';')[0]}")
    print()
    return payload, full, len(blob), mismatches


def threshold_demo():
    """change_threshold trades reconstruction fidelity for payload size.

    Uses an animation whose block changes are small (a growing cluster of
    pixels inside one 8x8 block), so thresholds below/above the cluster size
    demonstrably skip or record blocks.
    """
    frames = [_solid()]
    for i in range(1, N_FRAMES):
        nxt = frames[-1].copy()
        # i pixels change inside block (0, 0); cumulative, never reverted.
        for p in range(i):
            nxt[p // BLOCK, p % BLOCK] = (200, 200, 80)
        frames.append(nxt)

    print("### change_threshold sweep (sub-block changes inside one 8x8 block)")
    print(f"  changed pixels per delta frame grow 1..{N_FRAMES - 1}; "
          "a block is recorded only")
    print("  when its changed-pixel count is strictly greater than the threshold.")
    print(f"  {'threshold':>9} | {'payload (B)':>11} | "
          f"{'recorded blocks':>16} | {'mismatched pixels (all frames)':>32}")
    baseline_full = g.frame_diff_full_size(
        g.encode_frame_diff(frames, block_size=BLOCK)
    )
    for threshold in (0, 1, 2, 3, 5):
        seq = g.encode_frame_diff(frames, block_size=BLOCK,
                                  change_threshold=threshold)
        rebuilt = g.decode_frame_diff(seq)
        mismatch = sum(g.frame_diff_pixel_mismatch(frames, rebuilt))
        recorded = sum(g.frame_diff_changed_blocks(seq))
        print(f"  {threshold:9d} | {g.frame_diff_payload_size(seq):11d} | "
              f"{recorded:16d} | {mismatch:32d}")
    print(f"  full per-frame storage for reference: {baseline_full} B")
    print()


def main():
    print("=" * 72)
    print("Frame-difference (delta-block) encoding demo")
    print(f"Each animation: {N_FRAMES} frames, {WIDTH}x{HEIGHT} RGB "
          f"({WIDTH * HEIGHT * 3} raw bytes/frame)")
    print("=" * 72)
    print()

    animations = [
        ("1. Blink        -- one block toggles, everything else static", make_blink()),
        ("2. Sprite walk  -- one 8x8 block moves across the top band", make_sprite_walk()),
        ("3. Half scroll  -- lower half (8 of 16 blocks) changes/frame", make_half_scroll()),
        ("4. Full churn   -- every block changes every frame (worst case)", make_full_churn()),
    ]

    summary = []
    for name, frames in animations:
        summary.append((name, *report(name, frames)))

    threshold_demo()

    print("=" * 72)
    print("Summary (lossless payload bytes; full storage = "
          f"{N_FRAMES * WIDTH * HEIGHT * 3} B for every animation)")
    print("=" * 72)
    print(f"  {'animation':<52} {'delta':>7} {'full':>7} {'saving':>8}")
    for name, payload, full, _blob, _mis in summary:
        short = name.split(" -- ")[0]
        print(f"  {short:<52} {payload:7d} {full:7d} "
              f"{100.0 * (1 - payload / full):7.1f}%")
    print()
    print("All lossless reconstructions verified pixel-exact "
          "(0 differing pixels per frame).")


if __name__ == "__main__":
    main()
