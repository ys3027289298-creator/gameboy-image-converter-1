"""Small self-contained demo for gb_pipeline.dither_green_ramp.

Builds a tiny 32x32 green-toned sample in memory (no fixture files, no
network), quantizes it to four levels, runs both dither allocations at two
strengths, maps everything onto the 4-shade DMG green palette, prints the
four required metric groups, writes the small PNGs into artifacts/, and
exercises the invalid-parameter errors. Run from the repo root:

    /tmp/gbenv/bin/python scripts/green_dither_demo.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_pipeline as gp


def make_sample(width=32, height=32):
    """Deterministic green-toned sample: smooth diagonal ramp + soft blob.

    The smooth ramp is deliberately banding-prone after a 4-level direct
    quantization, so dithering has visible work to do. Green-dominant RGB
    keeps the result in the Game Boy green family.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    diagonal = (xx + yy) / (width + height - 2)
    blob = 0.25 * np.exp(-(((xx - 22) ** 2 + (yy - 10) ** 2) / 60.0))
    shade = np.clip(diagonal + blob, 0.0, 1.0)
    r = (30 + 150 * shade).astype(np.uint8)
    g = (60 + 180 * shade).astype(np.uint8)
    b = (40 + 120 * shade).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def _print_metrics(label, result):
    dist = ", ".join(f"{p * 100:6.2f}%" for p in result.level_distribution)
    print(f"\n=== {label} (method={result.method}, strength={result.strength}) ===")
    print(f"  1) level counts (L0..L3 light->dark): {result.level_counts}")
    print(f"     level distribution              : [{dist}]")
    print(
        "  2) mean luma before/after           : "
        f"{result.mean_luma_before:8.4f} / {result.mean_luma_after:8.4f}"
    )
    print(
        f"     mean luma bias (signed / |.|)   : "
        f"{result.mean_luma_bias:+8.4f} / {result.mean_abs_luma_bias:7.4f}"
    )
    print(
        "  3) MAE / RMSE vs source             : "
        f"{result.mae:8.4f} / {result.rmse:8.4f}"
    )
    print(
        "     direct quantization MAE / RMSE   : "
        f"{result.direct_mae:8.4f} / {result.direct_rmse:8.4f}"
    )
    print(
        "     delta vs direct (MAE / RMSE)     : "
        f"{result.mae_delta_vs_direct:+8.4f} / "
        f"{result.rmse_delta_vs_direct:+8.4f}"
    )
    print(
        "  4) repeat mismatches / consistent   : "
        f"{result.repeat_pixel_mismatches} / {result.consistent}"
    )


def main():
    sample = make_sample()
    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "green_dither"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.save(out_dir / "sample_source.png")
    print(f"sample: {sample.size[0]}x{sample.size[1]} RGB, PNGs -> {out_dir}")

    # Direct (undithered) quantization baseline via strength 0 on either path.
    direct = gp.dither_green_ramp(sample, method="bayer", strength=0.0)
    direct.image.save(out_dir / "sample_direct.png")
    _print_metrics("direct quantization (no dither)", direct)

    # External two-call consistency check in addition to the internal one.
    for method in gp.GREEN_DITHER_METHODS:
        for strength in (0.5, 1.0):
            result = gp.dither_green_ramp(sample, method=method, strength=strength)
            again = gp.dither_green_ramp(sample, method=method, strength=strength)
            external_match = np.array_equal(
                np.asarray(result.image), np.asarray(again.image)
            )
            _print_metrics(f"{method} dither", result)
            print(f"     external re-run pixel-identical : {external_match}")
            result.image.save(out_dir / f"sample_{method}_s{strength}.png")

    print("\n--- invalid parameter checks (each must raise) ---")
    bad_calls = [
        ("unknown method", {"method": "atkinson"}),
        ("strength > 1", {"method": "bayer", "strength": 1.5}),
        ("strength negative", {"method": "fs", "strength": -0.1}),
        ("strength NaN", {"method": "fs", "strength": float("nan")}),
        ("strength wrong type", {"method": "bayer", "strength": "1.0"}),
        ("bad ramp shape", {"method": "bayer", "ramp": np.zeros((3, 3), np.uint8)}),
        ("image wrong type", None),
    ]
    for label, kwargs in bad_calls:
        try:
            if kwargs is None:
                gp.dither_green_ramp(np.zeros((4, 4, 3), np.uint8))
            else:
                gp.dither_green_ramp(sample, **kwargs)
        except (TypeError, ValueError) as exc:
            print(f"  OK  {label:20s} -> {type(exc).__name__}: {exc}")
        else:
            raise AssertionError(f"expected error for {label}")


if __name__ == "__main__":
    main()
