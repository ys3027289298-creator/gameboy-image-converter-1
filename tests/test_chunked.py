"""Tests for the bounded-memory chunked converter (``convert_for_hardware_chunked``).

The chunked path splits only the per-pixel indexing/dithering work into
rectangular blocks while quantization, palette packing and per-tile palette
assignment stay GLOBAL, so every block shares one palette set. These tests pin
the four properties that makes that claim load-bearing:

* **pixel identity** -- for four input shapes (a single-block small image, a
  many-block large image, flat shapes deliberately straddling block seams, and
  a non-multiple-of-8 size that gets center-cropped), the stitched chunked
  output is byte-for-byte identical to the whole-image conversion, for both
  color and mono presets and both dither modes, with and without overlap;
* **peak memory** -- the chunked conversion's ``tracemalloc`` peak is strictly
  below the whole-image peak and shrinks monotonically as blocks get smaller
  (overlap pays back a bounded, measured amount);
* **runtime** -- each conversion is timed and reported (chunking must not
  introduce a pathological slowdown on these fixtures);
* **invalid block sizes** -- every malformed ``chunk_size`` /
  ``chunk_overlap`` raises ``ValueError`` before any conversion work happens.

All fixtures are tiny synthetic images generated deterministically here; no
network or third-party library beyond the repo's existing numpy/PIL stack is
used, and the whole file runs well inside the normal suite's budget.
"""

import time
import tracemalloc

import numpy as np
import pytest
from PIL import Image

import gb_pipeline as gp
from gb_pipeline import convert_for_hardware, convert_for_hardware_chunked


# ---------------------------------------------------------------------------
# Fixture factories -- small, deterministic, self-contained
# ---------------------------------------------------------------------------


def _photo_image(width, height, seed=23):
    """A deterministic gradient + mild noise RGB image at any size.

    Smooth ramps plus noise give the quantizer/indexer real tonal content
    (including Bayer-ditherable midtones) while staying tiny and reproducible.
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    r = (xx / max(width - 1, 1)) * 255.0
    g = (yy / max(height - 1, 1)) * 255.0
    b = ((xx + yy) / max(width + height - 2, 1)) * 255.0
    arr = np.stack([r, g, b], axis=-1) + rng.normal(0, 9, (height, width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _seam_crossing_image(width=64, height=64, seed=29):
    """Flat colored rectangles deliberately offset from every block seam.

    The 24px blocks used below have seams at x/y = 24 and 48; every rectangle
    below straddles at least one seam, so a block-local palette or stretch
    that drifted across a boundary would show up as a pixel difference.
    """
    rng = np.random.RandomState(seed)
    # Textured background so the canvas is not a single flat color.
    arr = rng.randint(0, 256, (height, width, 3)).astype(np.uint8)
    rectangles = [
        (10, 10, 30, 40, (220, 30, 30)),    # crosses the x=24 seam
        (20, 18, 52, 56, (30, 200, 40)),    # crosses both x=24 and x=48
        (4, 40, 60, 62, (40, 60, 220)),     # crosses the y=48 seam
        (44, 4, 62, 28, (230, 220, 40)),    # crosses y=48 and x=48
    ]
    for y0, x0, y1, x1, color in rectangles:
        arr[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _measure(fn):
    """Run ``fn`` under tracemalloc; return (result, peak_bytes, elapsed_s)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak, elapsed


def _assert_identical_conversion(image, chunk_kwargs):
    """Chunked ConversionResult equals the whole-image one pixel-for-pixel."""
    preset = chunk_kwargs["preset"]
    dither = chunk_kwargs.get("dither", "none")
    full = convert_for_hardware(image, preset, dither=dither)
    chunked = convert_for_hardware_chunked(image, **chunk_kwargs)
    assert np.array_equal(np.asarray(full.image), np.asarray(chunked.image)), (
        f"final render differs for {chunk_kwargs}"
    )
    assert np.array_equal(
        np.asarray(full.reference), np.asarray(chunked.reference)
    ), f"reference render differs for {chunk_kwargs}"
    assert full.stats == chunked.stats
    assert full.warnings == chunked.warnings
    return full, chunked


# Shared kwargs matrix: every preset/dither combination.
_PRESET_DITHER = [
    ("color_only", "none"),
    ("color_only", "bayer"),
    ("mono", "none"),
    ("mono", "bayer"),
]


# ---------------------------------------------------------------------------
# Category 1 -- small image: the whole canvas fits inside a single block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset,dither", _PRESET_DITHER)
def test_small_image_single_block_is_pixel_identical(preset, dither):
    # 16x16 = 2x2 tile cells; the 32x32 block covers the entire canvas at once,
    # so there is exactly one window and zero stitching boundaries.
    image = _photo_image(16, 16)
    full, chunked = _assert_identical_conversion(
        image,
        dict(preset=preset, dither=dither, chunk_size=(32, 32)),
    )
    assert chunked.image.size == full.image.size == (16, 16)


@pytest.mark.parametrize("preset,dither", _PRESET_DITHER)
def test_small_image_block_larger_than_canvas(preset, dither):
    # Blocks may exceed the canvas dimensions; windows simply clamp to it.
    image = _photo_image(24, 16)
    _assert_identical_conversion(
        image,
        dict(preset=preset, dither=dither, chunk_size=(128, 128)),
    )


# ---------------------------------------------------------------------------
# Category 2 -- large image: many blocks, stitched back in raster order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset,dither", _PRESET_DITHER)
def test_large_image_many_blocks_pixel_identical(preset, dither):
    # 64x64 over 16px blocks -> 4x4 = 16 ownership blocks, several windows.
    image = _photo_image(64, 64)
    full, chunked = _assert_identical_conversion(
        image,
        dict(preset=preset, dither=dither, chunk_size=(16, 16)),
    )
    # Every output pixel is RGB555-stable (hardware invariant still holds).
    out = np.asarray(chunked.image)
    assert np.array_equal(gp.snap_rgb555(out), out)


@pytest.mark.parametrize("preset,dither", _PRESET_DITHER)
def test_large_image_asymmetric_blocks_and_tile_counts(preset, dither):
    # Non-square, axis-uneven blocks (24x16) leave a ragged final column/row
    # (64 = 24 + 24 + 16), exercising the min() clamps in the window iterator.
    image = _photo_image(64, 64)
    _assert_identical_conversion(
        image,
        dict(preset=preset, dither=dither, chunk_size=(24, 16)),
    )


def test_block_windows_partition_the_tile_grid_in_raster_order():
    # Direct check of the window geometry: ownership rectangles must form an
    # exact, disjoint, in-order partition; processed rectangles extend past
    # them by the overlap (clamped at the canvas edges).
    tw = th = 8  # 8x8 tile-cell grid
    windows = list(gp._iter_chunk_windows(tw, th, 3, 2, 1, 1))
    owned = np.zeros((th, tw), dtype=np.uint8)
    for r0, c0, r1, c1, or0, oc0, or1, oc1 in windows:
        # Processed rectangle always contains the ownership rectangle.
        assert r0 <= or0 and or1 <= r1 and c0 <= oc0 and oc1 <= c1
        # Overlap never extends past the canvas.
        assert 0 <= r0 and r1 <= th and 0 <= c0 and c1 <= tw
        owned[or0:or1, oc0:oc1] += 1
    # Every cell owned exactly once.
    assert np.array_equal(owned, np.ones((th, tw), dtype=np.uint8))
    # Raster order: ownership origins sweep columns inside rows.
    origins = [(or0, oc0) for _, _, _, _, or0, oc0, _, _ in windows]
    assert origins == sorted(origins)


# ---------------------------------------------------------------------------
# Category 3 -- shapes crossing block boundaries (with and without overlap)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset,dither", _PRESET_DITHER)
@pytest.mark.parametrize("overlap", [(0, 0), (8, 8), (16, 8)])
def test_shapes_crossing_block_seams_stitch_cleanly(preset, dither, overlap):
    image = _seam_crossing_image()
    _assert_identical_conversion(
        image,
        dict(
            preset=preset,
            dither=dither,
            chunk_size=(24, 24),
            chunk_overlap=overlap,
        ),
    )


def test_overlap_windows_actually_extend_processed_rectangles():
    # With a one-cell overlap, interior windows process a rectangle larger than
    # their ownership rectangle; edge windows are clamped (no negative origin).
    windows = list(gp._iter_chunk_windows(tw=6, th=6, cw_t=2, ch_t=2,
                                          ow_t=1, oh_t=1))
    interior = [w for w in windows if w[0] > 0 and w[1] > 0
                and w[2] < 6 and w[3] < 6]
    assert interior, "expected at least one clamped-free interior window"
    for r0, c0, r1, c1, or0, oc0, or1, oc1 in interior:
        assert (r1 - r0) == (or1 - or0) + 2
        assert (c1 - c0) == (oc1 - oc0) + 2
    # First window touches the origin (clamped, no negative rows).
    assert windows[0][0] == 0 and windows[0][1] == 0


# ---------------------------------------------------------------------------
# Category 4 -- non-multiple-of-8 input dimensions (center-cropped)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset,dither", _PRESET_DITHER)
def test_non_divisible_size_cropped_then_stitched_identically(preset, dither):
    # 52x36 is not a multiple of 8; both paths center-crop to 48x32 first, so
    # the chunked output must still match the whole-image conversion exactly.
    image = _photo_image(52, 36)
    full, chunked = _assert_identical_conversion(
        image,
        dict(preset=preset, dither=dither, chunk_size=(16, 16)),
    )
    assert chunked.image.size == (48, 32)
    assert any("center-cropped" in w for w in chunked.warnings)
    assert chunked.warnings == full.warnings


def test_non_divisible_size_with_overlap():
    image = _photo_image(52, 36)
    _assert_identical_conversion(
        image,
        dict(
            preset="color_only",
            dither="bayer",
            chunk_size=(24, 16),
            chunk_overlap=(8, 8),
        ),
    )


# ---------------------------------------------------------------------------
# Peak memory -- chunked must stay below the whole-image peak
# ---------------------------------------------------------------------------


def _memory_image():
    # 96x96 photo content: large enough for per-block Lab tables to dominate
    # the traced peak, still cheap enough to convert repeatedly.
    return _photo_image(96, 96, seed=31)


def test_peak_memory_lower_and_monotonic_in_block_size():
    image = _memory_image()
    full, peak_full, t_full = _measure(
        lambda: convert_for_hardware(image, "color_only", dither="bayer")
    )
    peaks = {}
    timings = {}
    for csize in ((48, 48), (32, 32), (16, 16)):
        res, peak, elapsed = _measure(
            lambda c=csize: convert_for_hardware_chunked(
                image, "color_only", dither="bayer", chunk_size=c
            )
        )
        # Identity is re-asserted inside the memory run, not just assumed.
        assert np.array_equal(np.asarray(res.image), np.asarray(full.image))
        peaks[csize] = peak
        timings[csize] = elapsed

    # Every chunked peak is strictly below the whole-image peak.
    assert all(p < peak_full for p in peaks.values())
    # The smallest block's peak is at most half the whole-image peak -- the
    # bound the feature exists to deliver (measured ~0.15x on this fixture).
    assert peaks[(16, 16)] <= peak_full * 0.5
    # Monotone: shrinking blocks cannot raise the peak (allowing a tiny
    # measurement slack of one allocation quantum).
    assert peaks[(16, 16)] <= peaks[(32, 32)] + 4096
    assert peaks[(32, 32)] <= peaks[(48, 48)] + 4096
    # Timings are surfaced (printed with -s); chunking adds no pathological
    # cost -- the slowest chunked run stays within a generous 3x envelope.
    print(
        f"\npeak full={peak_full/1024:.1f}KiB t={t_full:.2f}s | "
        + " | ".join(
            f"chunk {c}: peak={peaks[c]/1024:.1f}KiB t={timings[c]:.2f}s"
            for c in ((48, 48), (32, 32), (16, 16))
        )
    )
    assert max(timings.values()) <= max(t_full * 3.0, 1.0)


def test_peak_memory_mono_and_overlap_overhead_is_bounded():
    image = _memory_image()
    _, peak_full, _ = _measure(
        lambda: convert_for_hardware(image, "mono", dither="bayer")
    )
    _, peak_no_overlap, _ = _measure(
        lambda: convert_for_hardware_chunked(
            image, "mono", dither="bayer", chunk_size=(24, 24)
        )
    )
    res_ov, peak_overlap, t_ov = _measure(
        lambda: convert_for_hardware_chunked(
            image, "mono", dither="bayer",
            chunk_size=(24, 24), chunk_overlap=(8, 8),
        )
    )
    full = convert_for_hardware(image, "mono", dither="bayer")
    assert np.array_equal(np.asarray(res_ov.image), np.asarray(full.image))
    assert peak_no_overlap < peak_full
    assert peak_overlap < peak_full
    # Overlap widens each window but must stay well under the full-image peak.
    assert peak_overlap <= peak_full * 0.8
    print(f"\nmono peaks: full={peak_full/1024:.1f}KiB "
          f"chunk={peak_no_overlap/1024:.1f}KiB "
          f"chunk+overlap={peak_overlap/1024:.1f}KiB t={t_ov:.2f}s")


def test_shared_palette_is_identical_across_blocks():
    # The whole point of global packing: every block must render from the SAME
    # palette stack. The stitched image's distinct colors must all be members
    # of that single palette set, including pixels at block seams.
    image = _seam_crossing_image()
    result = convert_for_hardware_chunked(
        image, "color_only", chunk_size=(24, 24), chunk_overlap=(8, 8)
    )
    out_colors = {
        tuple(c)
        for c in np.unique(np.asarray(result.image).reshape(-1, 3), axis=0)
    }
    # Exact palette equality with the whole-image conversion is the strongest
    # shared-palette guarantee (global packing, identical math).
    full = convert_for_hardware(image, "color_only")
    assert result.palette_hex == full.palette_hex
    # And every emitted color is one of those palette entries.
    allowed = {
        tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
        for pal in result.palette_hex for h in pal
    }
    assert out_colors <= allowed


# ---------------------------------------------------------------------------
# Invalid block sizes -- ValueError before any work, no partial output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(chunk_size=(0, 16)),                 # non-positive block
        dict(chunk_size=(16, 0)),
        dict(chunk_size=(-8, 16)),               # negative
        dict(chunk_size=(16, 7)),                # not a multiple of 8
        dict(chunk_size=(7, 16)),
        dict(chunk_overlap=(8, 7)),              # overlap not multiple of 8
        dict(chunk_size=(16, 16), chunk_overlap=(16, 0)),   # overlap == block
        dict(chunk_size=(8, 8), chunk_overlap=(16, 0)),     # overlap > block
        dict(chunk_size=(16, 16), chunk_overlap=(0, 16)),   # vertical overlap == block
        dict(chunk_overlap=(8, -8)),             # negative overlap
        dict(chunk_size=(16, 16.0)),             # float rejected, no truncation
        dict(chunk_size=(True, 16)),             # bool rejected
        dict(chunk_size=(16,)),                  # malformed tuple
        dict(chunk_overlap="16,8"),              # wrong type entirely
    ],
)
def test_invalid_chunk_parameters_raise_valueerror(kwargs):
    image = _photo_image(64, 64)
    with pytest.raises(ValueError):
        convert_for_hardware_chunked(image, "color_only", **kwargs)


def test_invalid_chunk_message_names_offending_field():
    image = _photo_image(64, 64)
    with pytest.raises(ValueError, match="chunk_h"):
        convert_for_hardware_chunked(image, "color_only", chunk_size=(16, 7))
    with pytest.raises(ValueError, match="overlap_w"):
        convert_for_hardware_chunked(
            image, "color_only", chunk_size=(8, 8), chunk_overlap=(16, 0)
        )


def test_logo_preset_rejected_from_chunked_path():
    # Logo canvases are fixed 160x144 and have no large-image memory problem.
    image = _photo_image(64, 64)
    with pytest.raises(ValueError, match="fixed canvas size"):
        convert_for_hardware_chunked(image, "logo_color")
    with pytest.raises(ValueError, match="fixed canvas size"):
        convert_for_hardware_chunked(image, "logo_mono")


def test_unknown_preset_and_dither_still_validated():
    image = _photo_image(64, 64)
    with pytest.raises(ValueError, match="unknown preset"):
        convert_for_hardware_chunked(image, "not_a_preset")
    with pytest.raises(ValueError, match="unknown dither"):
        convert_for_hardware_chunked(image, "mono", dither="floyd")


def test_numpy_integer_sizes_are_accepted():
    # numpy scalars are first-class sizes (callers compute them from arrays).
    image = _photo_image(32, 32)
    result = convert_for_hardware_chunked(
        image,
        "color_only",
        chunk_size=(np.int64(16), np.int64(16)),
        chunk_overlap=(np.int32(0), np.int32(0)),
    )
    full = convert_for_hardware(image, "color_only")
    assert np.array_equal(np.asarray(result.image), np.asarray(full.image))


def test_default_block_size_runs_and_matches():
    # The library defaults must be usable with no block arguments at all.
    image = _photo_image(48, 32)
    result = convert_for_hardware_chunked(image, "color_only")
    full = convert_for_hardware(image, "color_only")
    assert np.array_equal(np.asarray(result.image), np.asarray(full.image))
