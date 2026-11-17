"""Small deterministic sample images + custom pipeline combos.

Shared by ``scripts/make_pipeline_goldens.py`` (golden generation),
``tests/test_pipeline.py`` (pixel-identity + combo tests) and
``scripts/pipeline_demo.py`` (call graph / config / combo report), so all
three always see byte-identical inputs. Everything here is tiny (<=64x48)
and seeded, so the whole pipeline suite runs in one go.
"""

import copy

import numpy as np
from PIL import Image


def gradient_noise_sample(width: int = 64, height: int = 48, seed: int = 7) -> Image.Image:
    """64x48 RGB: 3-channel gradients + two saturated blocks + gaussian noise.

    Smooth ramps give the Bayer-dither stage something to dither; the saturated
    blocks give the palette packer distinct regions; the noise keeps the
    working-set quantizer honest.
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    r = xx / (width - 1) * 255
    g = yy / (height - 1) * 255
    b = (xx + yy) / (width + height - 2) * 255
    arr = np.stack([r, g, b], axis=-1)
    arr[8:24, 8:24] = (200, 60, 40)
    arr[24:40, 32:56] = (40, 80, 200)
    noise = rng.normal(0, 10, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), "RGB")


def flat_art_sample(width: int = 48, height: int = 48) -> Image.Image:
    """48x48 RGB: a few flat color regions with hard diagonal edges."""
    yy, xx = np.mgrid[0:height, 0:width]
    arr = np.full((height, width, 3), (240, 230, 200), dtype=np.uint8)
    arr[yy > xx] = (60, 140, 90)
    arr[(yy > 2 * xx // 3 + 10) & (xx > 24)] = (180, 50, 60)
    arr[6:18, 26:38] = (50, 60, 160)
    return Image.fromarray(arr, "RGB")


SAMPLES = {
    "grad": gradient_noise_sample,
    "flat": flat_art_sample,
}

# Golden cases: every preset x dither x sample. Logo presets resize their
# input to 160x144 internally, so the small samples still exercise them.
GOLDEN_CASES = [
    (img, preset, dither)
    for img in SAMPLES
    for preset in ("color_only", "mono", "logo_color", "logo_mono")
    for dither in ("none", "bayer")
]


def golden_case_name(img: str, preset: str, dither: str) -> str:
    return f"{img}__{preset}__{dither}"


def _default_stages() -> list:
    """A fresh (deep-copied) stage list matching the default pipeline."""
    return [
        {"stage": "prepare"},
        {"stage": "quantize"},
        {"stage": "palette_map"},
        {"stage": "dither"},
        {"stage": "dedup"},
        {"stage": "merge_budget"},
        {"stage": "conform"},
        {"stage": "export"},
    ]


def _set_stage(stages: list, name: str, **updates) -> None:
    for entry in stages:
        if entry["stage"] == name:
            entry.update(updates)
            return
    raise KeyError(name)


def custom_combos() -> dict:
    """The four custom pipeline combos exercised by the tests and the demo.

    Each value is (config, convert_kwargs). ``grad``/``flat`` samples are
    chosen per combo by the caller.
    """
    # 1. Force Bayer dithering from config (caller passes dither="none").
    bayer = _default_stages()
    _set_stage(bayer, "dither", params={"mode": "bayer"})

    # 2. Skip the working-set quantizer: pack palettes straight from the
    #    raw pixels.
    no_quant = _default_stages()
    _set_stage(no_quant, "quantize", enabled=False)

    # 3. Skip the lossy budget merge: tiles are never merged down to the
    #    budget, so a tight budget is reported as exceeded (honest warning).
    no_merge = _default_stages()
    _set_stage(no_merge, "merge_budget", enabled=False)

    # 4. Fast preview: no GB Studio oracle loop + forced Bayer dither.
    preview = _default_stages()
    _set_stage(preview, "conform", params={"oracle": False})
    _set_stage(preview, "dither", params={"mode": "bayer"})

    return {
        "bayer_from_config": ({"stages": bayer}, {"dither": "none"}),
        "no_quantize": ({"stages": no_quant}, {}),
        "no_merge_budget": ({"stages": no_merge}, {"tile_budget": 16}),
        "fast_preview": ({"stages": preview}, {}),
    }
