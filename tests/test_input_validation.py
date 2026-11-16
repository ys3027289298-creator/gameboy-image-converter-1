"""Tests for the input-validation layer in gb_pipeline.py.

User-supplied assets routinely arrive with non-tile-aligned dimensions,
grayscale/alpha channels, bilevel bit depths, or oversized embedded palettes.
``validate_input`` reports four per-item checks (dimensions, channels, bit
depth, palette colors) with pass/fail verdicts and remediation advice;
``autofix_input`` applies that remediation. The tests cover the four headline
input classes -- compliant, misaligned dimensions, grayscale, alpha -- plus
the 1-bit and over-limit-palette cases, and pin two guarantees:

* the auto-fixed image converts to the SAME pixels as feeding the original
  straight into ``convert_for_hardware`` (the fix mirrors the converter's own
  RGB-convert + center-crop preprocessing, so the original conversion output
  stays pixel-for-pixel unchanged);
* strict mode rejects every non-conforming input with
  ``InputValidationError`` carrying the same report data.

All fixtures are tiny (8-17 px) in-memory PIL images generated with seeded
numpy, so the whole module runs in seconds.
"""

import numpy as np
import pytest
from PIL import Image

from gb_pipeline import (
    INPUT_PALETTE_COLOR_LIMIT,
    TILE_SIZE,
    VALIDATION_AUTOFIX,
    VALIDATION_STRICT,
    InputValidationError,
    autofix_input,
    convert_for_hardware,
    validate_input,
)

CHECK_NAMES = ["dimensions", "channels", "bit_depth", "palette_colors"]


# ---------------------------------------------------------------------------
# Fixture builders -- small, deterministic, in-memory
# ---------------------------------------------------------------------------


def _rgb_image(width, height, seed=7):
    """A small deterministic RGB image: four flat colored quadrants + noise."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[: height // 2, : width // 2] = (200, 60, 60)
    arr[: height // 2, width // 2 :] = (60, 200, 60)
    arr[height // 2 :, : width // 2] = (60, 60, 200)
    arr[height // 2 :, width // 2 :] = (200, 200, 60)
    noise = rng.normal(0, 4, arr.shape)
    return Image.fromarray(
        np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8), "RGB"
    )


def _grayscale_image(width=16, height=16):
    return _rgb_image(width, height).convert("L")


def _alpha_image(width=16, height=16):
    """RGBA image whose alpha is a left-to-right ramp (flattening is visible)."""
    arr = np.asarray(_rgb_image(width, height).convert("RGBA")).copy()
    alpha = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    arr[..., 3] = alpha
    return Image.fromarray(arr, "RGBA")


def _bilevel_image(width=16, height=16):
    return _rgb_image(width, height).convert("1")


def _palette_image(width=16, height=16, colors=64):
    return _rgb_image(width, height).quantize(colors=colors)


@pytest.fixture
def compliant_image():
    return _rgb_image(16, 16)


@pytest.fixture
def odd_sized_image():
    # 17x13 -> center-cropped to 16x8 (whole 8x8 tiles).
    return _rgb_image(17, 13)


@pytest.fixture
def grayscale_image():
    return _grayscale_image()


@pytest.fixture
def alpha_image():
    return _alpha_image()


# ---------------------------------------------------------------------------
# Report structure -- the four checks are always present and well-formed
# ---------------------------------------------------------------------------


def test_report_always_has_four_named_checks(compliant_image):
    report = validate_input(compliant_image)
    assert [c.name for c in report.checks] == CHECK_NAMES
    assert report.passed is True
    assert report.mode == VALIDATION_AUTOFIX
    assert report.original_size == (16, 16)
    assert report.original_mode == "RGB"
    assert report.failures == []
    for check in report.checks:
        assert isinstance(check.passed, bool)
        assert check.detail  # every check reports a measurement
        if check.passed:
            assert check.fix == ""


def test_report_to_dict_roundtrip(compliant_image):
    report = validate_input(compliant_image).to_dict()
    assert report["passed"] is True
    assert [c["name"] for c in report["checks"]] == CHECK_NAMES
    for check in report["checks"]:
        assert set(check) == {"name", "passed", "detail", "fix", "applicable"}


def test_format_text_marks_every_verdict(compliant_image, odd_sized_image):
    text_ok = validate_input(compliant_image).format_text()
    assert "PASS" in text_ok
    for name in CHECK_NAMES:
        assert name in text_ok

    text_fail = validate_input(odd_sized_image).format_text()
    assert "FAIL" in text_fail
    assert "center-crop to 16x8" in text_fail
    # A failing check carries an actionable fix line.
    assert "fix:" in text_fail


def test_unknown_validation_mode_rejected(compliant_image):
    with pytest.raises(ValueError, match="unknown validation mode"):
        validate_input(compliant_image, mode="laissez-faire")


def test_validate_requires_pil_image():
    with pytest.raises(TypeError):
        validate_input(np.zeros((8, 8, 3), dtype=np.uint8))


def test_validation_does_not_mutate_input(alpha_image, odd_sized_image):
    for img in (alpha_image, odd_sized_image):
        before_mode, before_size = img.mode, img.size
        before_bytes = np.asarray(img).tobytes()
        validate_input(img)
        assert img.mode == before_mode
        assert img.size == before_size
        assert np.asarray(img).tobytes() == before_bytes


# ---------------------------------------------------------------------------
# The four headline input classes -- per-item verdicts and fix advice
# ---------------------------------------------------------------------------


def test_compliant_input_passes_all_checks(compliant_image):
    report = validate_input(compliant_image)
    assert report.passed is True
    assert report.get_check("dimensions").passed
    assert report.get_check("channels").passed
    assert report.get_check("bit_depth").passed
    # RGB images have no embedded palette: the check is vacuously applicable.
    palette_check = report.get_check("palette_colors")
    assert palette_check.passed and palette_check.applicable is False


def test_odd_sized_input_fails_only_dimensions(odd_sized_image):
    report = validate_input(odd_sized_image)
    assert report.passed is False
    assert [c.name for c in report.failures] == ["dimensions"]

    dims = report.get_check("dimensions")
    assert not dims.passed and dims.applicable
    assert "17x13" in dims.detail and "multiples of 8" in dims.detail
    assert "center-crop to 16x8" in dims.fix

    for name in ("channels", "bit_depth"):
        assert report.get_check(name).passed
    assert not report.get_check("palette_colors").applicable


def test_grayscale_input_fails_channels_only(grayscale_image):
    report = validate_input(grayscale_image)
    assert report.passed is False
    assert [c.name for c in report.failures] == ["channels"]

    channels = report.get_check("channels")
    assert "1 channel" in channels.detail and "grayscale" in channels.detail
    assert "convert to RGB" in channels.fix
    # An 8-bit grayscale image is still 8-bit per sample.
    assert report.get_check("bit_depth").passed
    assert report.get_check("dimensions").passed


def test_alpha_input_fails_channels_with_alpha_advice(alpha_image):
    report = validate_input(alpha_image)
    assert report.passed is False
    assert [c.name for c in report.failures] == ["channels"]

    channels = report.get_check("channels")
    assert "4 channels" in channels.detail and "alpha" in channels.detail
    assert "white" in channels.fix  # auto-fix documents its white matte
    assert report.get_check("dimensions").passed
    assert report.get_check("bit_depth").passed


def test_bilevel_input_fails_channels_and_bit_depth():
    report = validate_input(_bilevel_image())
    assert {c.name for c in report.failures} == {"channels", "bit_depth"}
    bit_depth = report.get_check("bit_depth")
    assert "1-bit" in bit_depth.detail and "8-bit" in bit_depth.fix


@pytest.mark.parametrize("mode", ["I", "F"])
def test_32bit_grayscale_fails_channels_and_bit_depth(mode):
    image = _grayscale_image().convert(mode)
    report = validate_input(image)
    assert {c.name for c in report.failures} == {"channels", "bit_depth"}
    assert "32-bit" in report.get_check("bit_depth").detail
    fixed, _ = autofix_input(image)
    assert fixed.mode == "RGB" and fixed.size == (16, 16)


def test_oversized_palette_fails_palette_count():
    image = _palette_image(colors=64)
    report = validate_input(image)
    palette_check = report.get_check("palette_colors")
    assert not palette_check.passed and palette_check.applicable
    assert "64" in palette_check.detail
    assert str(INPUT_PALETTE_COLOR_LIMIT) in palette_check.detail
    assert "decode to RGB" in palette_check.fix
    # A palette-indexed image also needs its channels decoded.
    assert not report.get_check("channels").passed


def test_palette_at_hardware_limit_passes():
    image = _palette_image(colors=INPUT_PALETTE_COLOR_LIMIT)
    report = validate_input(image)
    assert report.get_check("palette_colors").passed


def test_small_palette_reports_declared_count_not_padded_256():
    # Regression guard: Pillow pads getpalette() to 256 triples, so the check
    # must read the actually allocated palette (8 colors), not the flat list.
    image = _palette_image(colors=8)
    palette_check = validate_input(image).get_check("palette_colors")
    assert palette_check.passed
    assert "8 declared" in palette_check.detail


# ---------------------------------------------------------------------------
# Auto-fix -- output sizes and pixel-identical conversion parity
# ---------------------------------------------------------------------------


def _converted_pixels(image):
    return np.asarray(convert_for_hardware(image, "color_only").image)


def test_autofix_compliant_input_is_unchanged(compliant_image):
    fixed, report = autofix_input(compliant_image)
    assert report.passed
    assert fixed.size == (16, 16) and fixed.mode == "RGB"
    assert np.array_equal(np.asarray(fixed), np.asarray(compliant_image))


def test_autofix_odd_sized_crops_to_tile_grid(odd_sized_image):
    fixed, report = autofix_input(odd_sized_image)
    assert fixed.size == (16, 8)  # 17 -> 16, 13 -> 8, center-anchored
    assert fixed.mode == "RGB"
    assert not report.get_check("dimensions").passed
    # The cropped result is exactly the converter's own centered crop.
    direct = odd_sized_image.crop((0, (13 - 8) // 2, 16, (13 - 8) // 2 + 8))
    assert np.array_equal(np.asarray(fixed), np.asarray(direct))


def test_autofix_grayscale_promotes_to_rgb(grayscale_image):
    fixed, _report = autofix_input(grayscale_image)
    assert fixed.size == (16, 16) and fixed.mode == "RGB"
    expected = grayscale_image.convert("RGB")
    assert np.array_equal(np.asarray(fixed), np.asarray(expected))


def test_autofix_alpha_flattens_to_rgb(alpha_image):
    fixed, _report = autofix_input(alpha_image)
    assert fixed.size == (16, 16) and fixed.mode == "RGB"
    # PIL's documented default: composite over an opaque white background.
    expected = alpha_image.convert("RGB")
    assert np.array_equal(np.asarray(fixed), np.asarray(expected))


def test_autofix_bilevel_promotes_without_crop():
    image = _bilevel_image()
    fixed, _ = autofix_input(image)
    assert fixed.size == (16, 16) and fixed.mode == "RGB"


def test_autofix_oversized_palette_decodes_to_rgb():
    image = _palette_image(colors=64)
    fixed, report = autofix_input(image)
    assert fixed.mode == "RGB" and fixed.size == (16, 16)
    assert np.array_equal(np.asarray(fixed), np.asarray(image.convert("RGB")))
    assert not report.get_check("palette_colors").passed


def test_autofix_rejects_sub_tile_image():
    tiny = _rgb_image(7, 7)
    with pytest.raises(InputValidationError) as excinfo:
        autofix_input(tiny)
    assert not excinfo.value.report.passed
    assert not excinfo.value.report.get_check("dimensions").passed


@pytest.mark.parametrize(
    "fixture_name",
    ["compliant_image", "odd_sized_image", "grayscale_image", "alpha_image"],
)
def test_autofix_conversion_matches_direct_conversion_pixel_for_pixel(
    request, fixture_name
):
    """The four headline classes: fixed-path output == original-path output.

    This is the no-regression guarantee: because autofix only repeats the
    converter's own RGB conversion and centered tile crop, the diff count
    against today's ``convert_for_hardware`` output must be exactly zero.
    """
    image = request.getfixturevalue(fixture_name)
    direct = _converted_pixels(image)

    fixed, report = autofix_input(image)
    assert fixed.size[0] % TILE_SIZE == 0
    assert fixed.size[1] % TILE_SIZE == 0
    via_fix = _converted_pixels(fixed)

    assert direct.shape == via_fix.shape
    diff_count = int(np.sum(direct != via_fix))
    assert diff_count == 0
    # The auto-fix report is the input-state report, not a re-validation.
    assert report.original_size == image.size


def test_autofix_report_attached_in_strict_style_error(odd_sized_image):
    with pytest.raises(InputValidationError) as excinfo:
        validate_input(odd_sized_image, mode=VALIDATION_STRICT)
    report = excinfo.value.report
    assert report.mode == VALIDATION_STRICT
    assert not report.passed
    assert "dimensions" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Strict mode -- every non-conforming class is rejected, compliant accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name",
    ["odd_sized_image", "grayscale_image", "alpha_image"],
)
def test_strict_mode_rejects_nonconforming_inputs(request, fixture_name):
    image = request.getfixturevalue(fixture_name)
    with pytest.raises(InputValidationError) as excinfo:
        validate_input(image, mode=VALIDATION_STRICT)
    report = excinfo.value.report
    assert report.passed is False
    assert report.failures  # at least one failed check with advice
    for check in report.failures:
        assert check.fix


def test_strict_mode_accepts_compliant_input(compliant_image):
    report = validate_input(compliant_image, mode=VALIDATION_STRICT)
    assert report.passed is True
    assert report.mode == VALIDATION_STRICT


def test_strict_mode_rejects_bilevel_and_big_palette():
    with pytest.raises(InputValidationError):
        validate_input(_bilevel_image(), mode=VALIDATION_STRICT)
    with pytest.raises(InputValidationError):
        validate_input(_palette_image(colors=64), mode=VALIDATION_STRICT)


def test_strict_mode_error_carries_full_report(alpha_image):
    with pytest.raises(InputValidationError) as excinfo:
        validate_input(alpha_image, mode=VALIDATION_STRICT)
    report = excinfo.value.report
    assert [c.name for c in report.checks] == CHECK_NAMES
    channels = report.get_check("channels")
    assert not channels.passed and "alpha" in channels.detail


# ---------------------------------------------------------------------------
# Combined defects -- one report surfaces every failed item at once
# ---------------------------------------------------------------------------


def test_combined_grayscale_misaligned_input():
    # Grayscale AND non-multiple-of-8 dimensions: both checks must fire.
    image = _grayscale_image(13, 11)
    report = validate_input(image)
    assert {c.name for c in report.failures} == {"dimensions", "channels"}

    fixed, _ = autofix_input(image)
    assert fixed.size == (8, 8) and fixed.mode == "RGB"
    # Still pixel-parity with the direct path.
    assert np.array_equal(_converted_pixels(fixed), _converted_pixels(image))


def test_combined_alpha_misaligned_input():
    image = _alpha_image(17, 9)
    report = validate_input(image)
    assert {c.name for c in report.failures} == {"dimensions", "channels"}

    fixed, _ = autofix_input(image)
    assert fixed.size == (16, 8) and fixed.mode == "RGB"
    expected = image.convert("RGB").crop(
        ((17 - 16) // 2, (9 - 8) // 2,
         (17 - 16) // 2 + 16, (9 - 8) // 2 + 8)
    )
    assert np.array_equal(np.asarray(fixed), np.asarray(expected))
    assert np.array_equal(_converted_pixels(fixed), _converted_pixels(image))


def test_module_constants_are_hardware_aligned():
    assert TILE_SIZE == 8
    assert INPUT_PALETTE_COLOR_LIMIT == 32  # 8 palettes x 4 colors
    assert VALIDATION_STRICT != VALIDATION_AUTOFIX
