"""Tests for the sprite atlas packer in gb_pipeline.

Covers both layouts ("rows" and "maxrects"): 8x8-aligned placements,
recorded offsets/sizes, pixel-exact reconstruction of every sprite from the
atlas, configurable atlas size/alignment, and the not-fitting error path.
"""

import numpy as np
import pytest
from PIL import Image

from gb_pipeline import (
    AtlasPackError,
    pack_atlas,
    extract_from_atlas,
)


def make_sprite(width, height, seed):
    """Small deterministic RGBA sprite with a distinct per-seed pattern."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(height, width, 4), dtype=np.uint8)
    arr[..., 3] = 255
    return Image.fromarray(arr, mode="RGBA")


# Five small assets; sizes deliberately not all multiples of 8 so the
# alignment rounding is exercised.
FIVE_SPRITES = [
    ("banner", make_sprite(40, 16, 1)),
    ("pillar", make_sprite(16, 40, 2)),
    ("hero", make_sprite(24, 24, 3)),
    ("coin", make_sprite(8, 15, 4)),
    ("rock", make_sprite(16, 8, 5)),
]


@pytest.mark.parametrize("method", ["rows", "maxrects"])
def test_placements_are_aligned_and_non_overlapping(method):
    atlas = pack_atlas([im for _, im in FIVE_SPRITES],
                       names=[n for n, _ in FIVE_SPRITES],
                       size=(64, 64), align=8, method=method)
    assert atlas.image.size == (64, 64)
    occupied = np.zeros((64, 64), dtype=bool)
    for entry in atlas.entries:
        assert entry.x % 8 == 0 and entry.y % 8 == 0
        assert entry.x + entry.width <= 64 and entry.y + entry.height <= 64
        region = occupied[entry.y:entry.y + entry.height,
                          entry.x:entry.x + entry.width]
        assert not region.any(), f"{entry.name} overlaps another sprite"
        region[:] = True


@pytest.mark.parametrize("method", ["rows", "maxrects"])
def test_reconstruction_is_pixel_exact(method):
    atlas = pack_atlas([im for _, im in FIVE_SPRITES],
                       names=[n for n, _ in FIVE_SPRITES],
                       size=(64, 64), align=8, method=method)
    for (name, src), entry in zip(FIVE_SPRITES, atlas.entries):
        assert entry.name == name
        assert (entry.width, entry.height) == src.size
        rebuilt = extract_from_atlas(atlas.image, entry)
        assert rebuilt.tobytes() == src.convert("RGBA").tobytes()
        # Same thing via the name-based accessor.
        assert atlas.extract(name).tobytes() == rebuilt.tobytes()


@pytest.mark.parametrize("method", ["rows", "maxrects"])
def test_auto_size_and_utilization(method):
    atlas = pack_atlas([im for _, im in FIVE_SPRITES],
                       names=[n for n, _ in FIVE_SPRITES], method=method)
    w, h = atlas.size
    assert w % 8 == 0 and h % 8 == 0
    used = sum(im.width * im.height for _, im in FIVE_SPRITES)
    assert atlas.utilization() == pytest.approx(used / (w * h))
    assert 0.0 < atlas.utilization() <= 1.0


@pytest.mark.parametrize("method", ["rows", "maxrects"])
def test_fixed_size_too_small_raises(method):
    with pytest.raises(AtlasPackError):
        pack_atlas([im for _, im in FIVE_SPRITES], size=(16, 16), method=method)


def test_alignment_is_configurable():
    atlas = pack_atlas([im for _, im in FIVE_SPRITES],
                       names=[n for n, _ in FIVE_SPRITES],
                       size=(80, 80), align=16, method="maxrects")
    for entry in atlas.entries:
        assert entry.x % 16 == 0 and entry.y % 16 == 0


def test_atlas_size_must_respect_alignment():
    with pytest.raises(ValueError):
        pack_atlas([FIVE_SPRITES[0][1]], size=(20, 20), align=8)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        pack_atlas([FIVE_SPRITES[0][1]], method="spiral")


def test_empty_input_raises():
    with pytest.raises(ValueError):
        pack_atlas([])


def test_entries_follow_input_order_not_layout_order():
    # A tall-first sprite list would be reordered internally; entries must
    # still line up with the input order.
    sprites = [make_sprite(8, 40, 11), make_sprite(40, 8, 12), make_sprite(8, 8, 13)]
    atlas = pack_atlas(sprites, names=["tall", "wide", "tiny"], size=(64, 64),
                       method="rows")
    assert [e.name for e in atlas.entries] == ["tall", "wide", "tiny"]
    for src, entry in zip(sprites, atlas.entries):
        rebuilt = extract_from_atlas(atlas.image, entry)
        assert rebuilt.tobytes() == src.tobytes()
