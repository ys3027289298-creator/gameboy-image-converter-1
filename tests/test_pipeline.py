"""Tests for the composable pipeline engine (DEFAULT_PIPELINE_CONFIG +
convert_with_pipeline).

The golden fixtures under ``fixtures/pipeline_golden/`` were generated from
the pre-refactor, hardcoded ``convert_for_hardware`` (commit 4f14632) by
``scripts/make_pipeline_goldens.py``. Comparing the config-driven engine
against them pixel by pixel is the evidence that the default pipeline config
reproduces the old behavior exactly.
"""

import copy
import json
import os

import numpy as np
import pytest
from PIL import Image

import gb_pipeline
from gb_pipeline import (
    DEFAULT_PIPELINE_CONFIG,
    PipelineConfigError,
    convert_for_hardware,
    convert_with_pipeline,
    describe_pipeline,
)
from tests.pipeline_samples import (
    GOLDEN_CASES,
    SAMPLES,
    custom_combos,
    golden_case_name,
)

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "pipeline_golden")


def _rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


# -- Default config == pre-refactor behavior, pixel by pixel ------------------


@pytest.mark.parametrize("img_name,preset,dither", GOLDEN_CASES)
def test_default_config_matches_pre_refactor_golden(img_name, preset, dither):
    """convert_for_hardware (now engine-driven) must reproduce the golden
    final image, reference image, stats, warnings, and palette hex exactly."""
    case = golden_case_name(img_name, preset, dither)
    res = convert_for_hardware(SAMPLES[img_name](), preset, dither=dither)

    golden = _rgb(Image.open(os.path.join(GOLDEN_DIR, case + ".png")))
    golden_ref = _rgb(Image.open(os.path.join(GOLDEN_DIR, case + "__ref.png")))
    with open(os.path.join(GOLDEN_DIR, case + ".json")) as fh:
        meta = json.load(fh)

    # Per-pixel evidence: zero differing pixels, zero max channel delta.
    diff = np.asarray(res.image.convert("RGB"), dtype=np.int16) - golden.astype(np.int16)
    assert int(np.abs(diff).max()) == 0, f"{case}: final image differs"
    diff_ref = np.asarray(res.reference.convert("RGB"), dtype=np.int16) - golden_ref.astype(np.int16)
    assert int(np.abs(diff_ref).max()) == 0, f"{case}: reference image differs"
    assert res.stats == meta["stats"]
    assert res.warnings == meta["warnings"]
    assert res.palette_hex == meta["palette_hex"]


def test_convert_with_pipeline_default_config_is_convert_for_hardware():
    """convert_with_pipeline(config=None) and an explicit copy of
    DEFAULT_PIPELINE_CONFIG both equal convert_for_hardware, pixel for pixel."""
    img = SAMPLES["grad"]()
    expected = convert_for_hardware(img, "color_only", dither="bayer")
    for config in (None, copy.deepcopy(DEFAULT_PIPELINE_CONFIG)):
        res = convert_with_pipeline(img, "color_only", config, dither="bayer")
        assert np.array_equal(_rgb(res.image), _rgb(expected.image))
        assert np.array_equal(_rgb(res.reference), _rgb(expected.reference))
        assert res.stats == expected.stats
        assert res.warnings == expected.warnings


def test_default_config_shape():
    """The default config lists the eight stages of the original flow."""
    assert [s["stage"] for s in DEFAULT_PIPELINE_CONFIG["stages"]] == [
        "prepare", "quantize", "palette_map", "dither",
        "dedup", "merge_budget", "conform", "export",
    ]


def test_describe_pipeline_call_graph():
    """The call graph exposes every stage in execution order with its edges."""
    graph = describe_pipeline()
    names = [s["name"] for s in graph["stages"]]
    assert names == [
        "prepare", "quantize", "palette_map", "dither",
        "dedup", "merge_budget", "conform", "export",
    ]
    edge_pairs = {(src, dst) for src, dst, _ in graph["edges"]}
    # The backbone of the original hardcoded flow.
    for expected in [
        ("prepare", "quantize"), ("prepare", "palette_map"),
        ("quantize", "palette_map"), ("palette_map", "dither"),
        ("dither", "dedup"), ("dedup", "merge_budget"),
        ("prepare", "conform"), ("conform", "export"),
    ]:
        assert expected in edge_pairs


# -- Custom combos ------------------------------------------------------------


def test_combo_bayer_from_config_overrides_caller_dither():
    """dither stage param mode=bayer wins over the caller's dither='none'."""
    img = SAMPLES["grad"]()
    config, kwargs = custom_combos()["bayer_from_config"]
    res = convert_with_pipeline(img, "color_only", config, **kwargs)
    default = convert_for_hardware(img, "color_only", dither="none")
    bayer = convert_for_hardware(img, "color_only", dither="bayer")
    assert res.stats["dither"] == "bayer"
    assert not np.array_equal(_rgb(res.image), _rgb(default.image))
    assert np.array_equal(_rgb(res.image), _rgb(bayer.image))


def test_combo_no_quantize_packs_raw_pixels():
    """Disabling the quantize stage changes the output but keeps it GB-legal
    (the export stage's round-trip verification still passes)."""
    img = SAMPLES["grad"]()
    config, kwargs = custom_combos()["no_quantize"]
    res = convert_with_pipeline(img, "color_only", config, **kwargs)
    default = convert_for_hardware(img, "color_only")
    assert not np.array_equal(_rgb(res.image), _rgb(default.image))
    assert res.stats["palettes_used"] <= 8


def test_combo_no_merge_budget_exceeds_budget_with_warning():
    """Disabling merge_budget leaves the tile count above a tight budget and
    surfaces the honest over-budget warning instead of enforcing it."""
    img = SAMPLES["grad"]()
    config, kwargs = custom_combos()["no_merge_budget"]
    res = convert_with_pipeline(img, "color_only", config, **kwargs)
    assert res.stats["n_merges"] == 0
    assert res.stats["tiles_used"] > 16
    assert any("exceeds the 16 budget" in w for w in res.warnings)


def test_combo_fast_preview_skips_oracle():
    """conform oracle=false ships a single candidate with no GB Studio
    importer loop; the result is still a structurally valid ConversionResult."""
    img = SAMPLES["grad"]()
    config, kwargs = custom_combos()["fast_preview"]
    res = convert_with_pipeline(img, "color_only", config, **kwargs)
    assert isinstance(res, gb_pipeline.ConversionResult)
    assert res.stats["tiles_used"] >= 1
    assert res.stats["dither"] == "bayer"
    # No oracle loop -> no GB Studio recoloring was needed or measured.
    assert res.stats["gbstudio_recolored_tiles"] == 0


# -- Config error reporting ---------------------------------------------------


def test_unknown_stage_error():
    config = {"stages": ["prepare", "frobulate", "export"]}
    with pytest.raises(PipelineConfigError, match="unknown pipeline stage") as exc:
        convert_with_pipeline(SAMPLES["flat"](), "color_only", config)
    msg = str(exc.value)
    assert "'frobulate'" in msg
    assert "Available stages" in msg and "quantize" in msg


def test_cyclic_dependency_error():
    """quantize after dither, while dither needs palette_map needs quantize."""
    config = {
        "stages": [
            "prepare",
            {"stage": "quantize", "after": ["dither"]},
            "palette_map", "dither", "dedup", "merge_budget", "conform", "export",
        ]
    }
    with pytest.raises(PipelineConfigError, match="cyclic pipeline dependency") as exc:
        convert_with_pipeline(SAMPLES["flat"](), "color_only", config)
    # The message spells out the concrete cycle.
    assert "quantize" in str(exc.value) and "->" in str(exc.value)


def test_missing_provider_error():
    """Disabling dither leaves dedup's 'gb' requirement unmet."""
    config = {
        "stages": [
            "prepare", "quantize", "palette_map",
            {"stage": "dither", "enabled": False},
            "dedup", "merge_budget", "conform", "export",
        ]
    }
    with pytest.raises(PipelineConfigError, match="requires 'gb'"):
        convert_with_pipeline(SAMPLES["flat"](), "color_only", config)


def test_duplicate_stage_error():
    config = {"stages": ["prepare", "quantize", "quantize"]}
    with pytest.raises(PipelineConfigError, match="duplicate pipeline stage"):
        convert_with_pipeline(SAMPLES["flat"](), "color_only", config)


def test_unknown_stage_param_error():
    config = {"stages": [{"stage": "dither", "params": {"speed": "fast"}}]}
    with pytest.raises(PipelineConfigError, match="unknown parameter"):
        convert_with_pipeline(SAMPLES["flat"](), "color_only", config)


def test_dangling_after_error():
    stages = [dict(s) for s in DEFAULT_PIPELINE_CONFIG["stages"]]
    stages[-1] = {"stage": "export", "after": ["nonexistent"]}
    with pytest.raises(PipelineConfigError, match="not an enabled stage"):
        convert_with_pipeline(SAMPLES["flat"](), "color_only", {"stages": stages})


def test_pipeline_config_error_is_value_error():
    """PipelineConfigError subclasses ValueError for existing callers."""
    assert issubclass(PipelineConfigError, ValueError)
