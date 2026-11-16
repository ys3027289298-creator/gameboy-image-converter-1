"""Tests for gb_pipeline's animation compositing and GIF export.

Covers: per-frame dispose semantics (keep / background / previous), the
first-frame-base toggle, shared-palette consistency, single-frame export
pixel-identity with the direct export, GIF metadata (frame count, per-frame
delays, loop count), and the mismatched-frame-size error.
"""

import io

import numpy as np
import pytest
from PIL import Image

import gb_pipeline as gp


def _rgba_frame(size, draws, base=None):
    """A small RGBA frame: uniform `base` (or transparent) plus (xy, rgba) draws."""
    img = Image.new("RGBA", size, base if base is not None else (0, 0, 0, 0))
    arr = np.asarray(img).copy()
    for (x, y), rgba in draws.items():
        arr[y, x] = rgba
    return Image.fromarray(arr, "RGBA")


def _px(img, x, y):
    return tuple(np.asarray(img)[y, x])


RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
BG = (10, 20, 30)  # RGB background used across the compositing tests


def test_compose_keep_accumulates_frames():
    f0 = _rgba_frame((4, 4), {(0, 0): RED}, base=BG + (255,))
    f1 = _rgba_frame((4, 4), {(1, 1): GREEN})
    f2 = _rgba_frame((4, 4), {(2, 2): BLUE})
    out, delays = gp.compose_animation(
        [f0, f1, f2], delays=[100, 200, 300], disposes="keep", background=BG
    )
    assert len(out) == 3 and delays == [100, 200, 300]
    # Frame 2 shows everything drawn so far.
    assert _px(out[2], 0, 0) == RED
    assert _px(out[2], 1, 1) == GREEN
    assert _px(out[2], 2, 2) == BLUE
    assert _px(out[2], 3, 3) == BG + (255,)


def test_compose_background_clears_drawn_pixels():
    f0 = _rgba_frame((4, 4), {(0, 0): RED}, base=BG + (255,))
    f1 = _rgba_frame((4, 4), {(1, 1): GREEN})
    f2 = _rgba_frame((4, 4), {})  # draws nothing
    out, _ = gp.compose_animation(
        [f0, f1, f2], disposes=["keep", "background", "keep"], background=BG
    )
    # f1's dispose=background cleared its drawn pixel, so f2 shows f0's state.
    assert _px(out[2], 1, 1) == BG + (255,)
    assert _px(out[2], 0, 0) == RED
    # But while f1 was up, its pixel was visible.
    assert _px(out[1], 1, 1) == GREEN


def test_compose_previous_restores_prior_canvas():
    f0 = _rgba_frame((4, 4), {(0, 0): RED}, base=BG + (255,))
    f1 = _rgba_frame((4, 4), {(1, 1): GREEN})
    f2 = _rgba_frame((4, 4), {})
    out, _ = gp.compose_animation(
        [f0, f1, f2], disposes=["keep", "previous", "keep"], background=BG
    )
    assert _px(out[2], 1, 1) == BG + (255,)  # f1 rolled back
    assert _px(out[2], 0, 0) == RED          # f0 (kept) still visible


def test_first_frame_base_flag_controls_frame0_dispose():
    f0 = _rgba_frame((4, 4), {(0, 0): RED})
    f1 = _rgba_frame((4, 4), {})
    # As baseline (default): frame 0's dispose=previous is ignored, it stays.
    out_base, _ = gp.compose_animation(
        [f0, f1], disposes=["previous", "keep"], background=BG,
        first_frame_base=True,
    )
    assert _px(out_base[1], 0, 0) == RED
    # As an ordinary frame: dispose=previous rolls it back to the background.
    out_plain, _ = gp.compose_animation(
        [f0, f1], disposes=["previous", "keep"], background=BG,
        first_frame_base=False,
    )
    assert _px(out_plain[1], 0, 0) == BG + (255,)


def test_compose_rejects_mismatched_frame_sizes():
    f0 = _rgba_frame((4, 4), {})
    f1 = _rgba_frame((5, 4), {})
    with pytest.raises(ValueError, match="frame sizes differ"):
        gp.compose_animation([f0, f1])


def test_compose_rejects_empty_and_bad_options():
    with pytest.raises(ValueError, match="at least one frame"):
        gp.compose_animation([])
    f0 = _rgba_frame((4, 4), {})
    with pytest.raises(ValueError, match="dispose"):
        gp.compose_animation([f0], disposes=["bogus"])
    with pytest.raises(ValueError, match="non-negative"):
        gp.compose_animation([f0], delays=[-1])
    with pytest.raises(ValueError, match="expected 2 entries"):
        gp.compose_animation([f0, f0], delays=[100])


def test_shared_palette_frames_are_palette_consistent():
    f0 = _rgba_frame((4, 4), {(0, 0): RED, (1, 0): GREEN}, base=BG + (255,))
    f1 = _rgba_frame((4, 4), {(2, 0): BLUE}, base=BG + (255,))
    out, pal = gp.shared_palette_frames([f0, f1])
    assert all(im.mode == "P" for im in out)
    assert out[0].getpalette() == out[1].getpalette()
    # Every pixel color of every frame lives in the shared palette.
    pal_colors = {tuple(pal[i]) for i in range(len(pal))}
    for im in out:
        used = {tuple(im.getpalette()[i * 3: i * 3 + 3]) for _, i in im.getcolors()}
        assert used <= pal_colors
    # Lossless: colors were few enough that the mapping is exact.
    assert np.array_equal(np.asarray(out[0].convert("RGB")), np.asarray(f0.convert("RGB")))


def test_shared_palette_frames_custom_palette_maps_to_members():
    ramp = np.array(
        [[0, 0, 0], [85, 85, 85], [170, 170, 170], [255, 255, 255]],
        dtype=np.uint8,
    )
    f0 = _rgba_frame((4, 4), {(0, 0): (250, 250, 250, 255)}, base=(8, 8, 8, 255))
    out, pal = gp.shared_palette_frames([f0], palette=ramp)
    used = {tuple(out[0].getpalette()[i * 3: i * 3 + 3]) for _, i in out[0].getcolors()}
    assert used <= {tuple(c) for c in ramp.tolist()}
    # Near-white snaps to white, near-black to black.
    assert _px(out[0].convert("RGB"), 0, 0) == (255, 255, 255)
    assert _px(out[0].convert("RGB"), 1, 1) == (0, 0, 0)


def _read_gif_frames(data):
    im = Image.open(io.BytesIO(data))
    frames, durations = [], []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append(im.convert("RGB").copy())
        durations.append(im.info.get("duration"))
    return frames, durations, im.info.get("loop")


def test_export_animation_gif_roundtrip_pixels_and_metadata():
    draws = [{(0, 0): RED}, {(1, 1): GREEN}, {(2, 2): BLUE}]
    frames = [_rgba_frame((8, 8), d, base=BG + (255,)) for d in draws]
    delays = [120, 200, 80]
    data, info = gp.export_animation(
        frames, delays=delays, disposes="keep", loop=3, background=BG
    )
    assert info["n_frames"] == 3
    assert info["delays_ms"] == delays
    assert info["total_duration_ms"] == 400
    assert info["loop"] == 3
    assert info["size_bytes"] == len(data)

    got, got_durations, got_loop = _read_gif_frames(data)
    assert got_loop == 3
    assert got_durations == delays
    composited, _ = gp.compose_animation(frames, delays=delays, background=BG)
    for got_frame, want in zip(got, composited):
        assert np.array_equal(
            np.asarray(got_frame), np.asarray(want.convert("RGB"))
        )


def test_export_animation_single_frame_pixel_identical():
    """The single-frame export is pixel-identical to the direct export."""
    rng = np.random.RandomState(7)
    # A GB-style image: few colors, well under the 256-color palette limit.
    colors = np.array(
        [[224, 248, 208], [136, 192, 112], [52, 104, 86], [8, 24, 32]],
        dtype=np.uint8,
    )
    arr = colors[rng.randint(0, 4, size=(32, 32))]
    src = Image.fromarray(arr, "RGB")

    data, info = gp.export_animation([src], delays=[100])
    assert info["n_frames"] == 1
    got, _, _ = _read_gif_frames(data)
    assert np.array_equal(np.asarray(got[0]), np.asarray(src))


def test_export_animation_rejects_bad_loop_and_mismatched_sizes():
    f0 = _rgba_frame((4, 4), {})
    with pytest.raises(ValueError, match="loop"):
        gp.export_animation([f0], loop=-1)
    with pytest.raises(ValueError, match="frame sizes differ"):
        gp.export_animation([f0, _rgba_frame((8, 8), {})])
