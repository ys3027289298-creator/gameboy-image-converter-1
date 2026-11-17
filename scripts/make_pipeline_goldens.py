"""Regenerate the pipeline golden fixtures from ``convert_for_hardware``.

Run from the repo root with the project venv:

    python scripts/make_pipeline_goldens.py

The goldens under ``tests/fixtures/pipeline_golden/`` were generated from the
pre-pipeline-engine implementation (commit 4f14632, the hardcoded
``convert_for_hardware`` body). ``tests/test_pipeline.py`` compares the
config-driven engine against them pixel by pixel -- that comparison is the
evidence that the default pipeline config reproduces the old behavior
exactly. Re-running this script on the refactored code should produce
byte-identical files.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import gb_pipeline  # noqa: E402
from tests.pipeline_samples import GOLDEN_CASES, SAMPLES, golden_case_name  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "pipeline_golden")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for img_name, preset, dither in GOLDEN_CASES:
        case = golden_case_name(img_name, preset, dither)
        res = gb_pipeline.convert_for_hardware(
            SAMPLES[img_name](), preset, dither=dither
        )
        res.image.save(os.path.join(OUT_DIR, case + ".png"))
        res.reference.save(os.path.join(OUT_DIR, case + "__ref.png"))
        with open(os.path.join(OUT_DIR, case + ".json"), "w") as fh:
            json.dump(
                {
                    "image": img_name,
                    "preset": preset,
                    "dither": dither,
                    "stats": res.stats,
                    "warnings": res.warnings,
                    "palette_hex": res.palette_hex,
                },
                fh,
                indent=2,
                sort_keys=True,
            )
        print(f"golden {case}: tiles={res.stats['tiles_used']} "
              f"palettes={res.stats['palettes_used']} merges={res.stats['n_merges']}")
    print(f"wrote {len(GOLDEN_CASES)} golden cases to {OUT_DIR}")


if __name__ == "__main__":
    main()
