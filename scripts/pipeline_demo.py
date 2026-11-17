"""Print the pipeline call graph, the default config, a four-combo output
comparison, and the config-error messages. Read-only demo; run from the repo
root:

    python scripts/pipeline_demo.py
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

import gb_pipeline  # noqa: E402
from gb_pipeline import (  # noqa: E402
    DEFAULT_PIPELINE_CONFIG,
    PipelineConfigError,
    convert_for_hardware,
    convert_with_pipeline,
    describe_pipeline,
)
from tests.pipeline_samples import SAMPLES, custom_combos  # noqa: E402


def show_call_graph() -> None:
    print("=" * 72)
    print("CALL GRAPH (default pipeline, execution order)")
    print("=" * 72)
    graph = describe_pipeline()
    for stage in graph["stages"]:
        outs = ", ".join(stage["provides"]) or "-"
        ins = ", ".join(stage["requires"]) or "-"
        print(f"  [{stage['group']:>8}] {stage['name']:<12} in: {ins:<28} out: {outs}")
    print("\n  edges (from -> to : why):")
    for src, dst, why in graph["edges"]:
        print(f"    {src:<12} -> {dst:<12} : {why}")
    print("\n  main flow:      prepare -> conform -> export")
    print("  candidate group (re-run per palette count, memoized):")
    print("                  quantize -> palette_map -> dither -> dedup")
    print("  merge group (re-run per tile target): merge_budget")


def show_default_config() -> None:
    print("\n" + "=" * 72)
    print("DEFAULT CONFIG (gb_pipeline.DEFAULT_PIPELINE_CONFIG)")
    print("=" * 72)
    print(json.dumps(DEFAULT_PIPELINE_CONFIG, indent=2))


def show_combos() -> None:
    print("\n" + "=" * 72)
    print("FOUR CUSTOM COMBOS vs DEFAULT (sample: 64x48 'grad', preset color_only)")
    print("=" * 72)
    img = SAMPLES["grad"]()
    base = convert_for_hardware(img, "color_only")
    base_arr = np.asarray(base.image.convert("RGB"), dtype=np.int16)
    print(f"\n  default: tiles={base.stats['tiles_used']} "
          f"palettes={base.stats['palettes_used']} merges={base.stats['n_merges']} "
          f"dither={base.stats['dither']} warnings={len(base.warnings)}")
    for name, (config, kwargs) in custom_combos().items():
        res = convert_with_pipeline(img, "color_only", config, **kwargs)
        arr = np.asarray(res.image.convert("RGB"), dtype=np.int16)
        n_diff = int(np.any(arr != base_arr, axis=2).sum())
        print(f"\n  combo {name}")
        print(f"    pixels differing vs default : {n_diff} / {arr.shape[0] * arr.shape[1]}")
        print(f"    tiles_used={res.stats['tiles_used']} "
              f"palettes_used={res.stats['palettes_used']} "
              f"n_merges={res.stats['n_merges']} dither={res.stats['dither']}")
        for w in res.warnings:
            print(f"    warning: {w}")


def show_errors() -> None:
    print("\n" + "=" * 72)
    print("CONFIG ERROR MESSAGES")
    print("=" * 72)
    img = SAMPLES["flat"]()
    cases = {
        "unknown stage": {"stages": ["prepare", "frobulate", "export"]},
        "cyclic dependency": {
            "stages": [
                "prepare",
                {"stage": "quantize", "after": ["dither"]},
                "palette_map", "dither", "dedup", "merge_budget",
                "conform", "export",
            ]
        },
        "unmet requirement (dither disabled)": {
            "stages": [
                "prepare", "quantize", "palette_map",
                {"stage": "dither", "enabled": False},
                "dedup", "merge_budget", "conform", "export",
            ]
        },
    }
    for label, config in cases.items():
        try:
            convert_with_pipeline(img, "color_only", config)
            print(f"\n  {label}: NO ERROR (unexpected)")
        except PipelineConfigError as exc:
            print(f"\n  {label}:\n    PipelineConfigError: {exc}")


if __name__ == "__main__":
    show_call_graph()
    show_default_config()
    show_combos()
    show_errors()
