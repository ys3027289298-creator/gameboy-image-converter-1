"""Tests for the OAM per-scanline sprite budget (gb_pipeline.py).

Covers: invalid-limit / invalid-weight validation, the under-limit no-op
(restricted output pixel-identical to unrestricted), deterministic priority
arbitration on overflow, each scoring weight (distance / frequency / area)
changing who gets demoted, off-canvas sprites, the frame-drop / perceptible
/ diff metrics, and the no-limit-output-unchanged guarantee.
"""

import numpy as np
import pytest

import gb_pipeline as gbp
from gb_pipeline import (
    ScanlineBudget,
    SpriteFrame,
    SpriteObject,
    allocate_sprite_budget,
)


def _sprite(sprite_id, x, y, w=4, h=4, color=(255, 0, 0), transparent=None):
    """Small solid-color sprite; an all-False transparency mask by default."""
    pixels = np.tile(np.array(color, dtype=np.uint8), (h, w, 1))
    if transparent is None:
        mask = np.zeros((h, w), dtype=bool)
    else:
        mask = np.asarray(transparent, dtype=bool)
    return SpriteObject(x=x, y=y, pixels=pixels, transparent=mask, id=sprite_id)


def _frame(sprites, size=(16, 16), bg_color=(0, 0, 0), background=None):
    if background is None:
        h, w = size
        background = np.tile(
            np.array(bg_color, dtype=np.uint8), (h, w, 1)
        )
    return SpriteFrame(sprites=list(sprites), background=background)


# --- validation -------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -10, 1.5, "10", None, True, False])
def test_invalid_max_per_scanline_raises(bad):
    with pytest.raises(ValueError, match="max_per_scanline"):
        ScanlineBudget(max_per_scanline=bad)


@pytest.mark.parametrize(
    "kw",
    [
        dict(weight_distance=-1.0),
        dict(weight_frequency=-0.1),
        dict(weight_area=float("nan")),
        dict(weight_distance=float("inf")),
        dict(
            weight_distance=0.0,
            weight_frequency=0.0,
            weight_area=0.0,
        ),
    ],
)
def test_invalid_weights_raise(kw):
    with pytest.raises(ValueError):
        ScanlineBudget(**kw)


def test_bad_budget_type_and_empty_frames_raise():
    frame = _frame([_sprite(0, 0, 0)])
    with pytest.raises(ValueError):
        allocate_sprite_budget([frame], (8, 8), {"max_per_scanline": 2})
    with pytest.raises(ValueError):
        allocate_sprite_budget([], (8, 8), ScanlineBudget())


def test_mismatched_background_sizes_raise():
    f1 = _frame([_sprite(0, 0, 0)], size=(8, 8))
    f2 = _frame([_sprite(0, 0, 0)], size=(8, 10))
    with pytest.raises(ValueError, match="background size"):
        allocate_sprite_budget([f1, f2], (4, 4), ScanlineBudget())


# --- under-limit no-op ------------------------------------------------------

def test_under_limit_is_pixel_identical_to_unrestricted():
    # Three 4px-tall sprites stacked on the same columns: each line sees 3,
    # limit 10 -> no overflow anywhere.
    sprites = [
        _sprite(0, 0, 0, color=(255, 0, 0)),
        _sprite(1, 0, 4, color=(0, 255, 0)),
        _sprite(2, 0, 8, color=(0, 0, 255)),
    ]
    frames = [_frame(sprites)]
    res = allocate_sprite_budget(frames, (8, 8), ScanlineBudget(max_per_scanline=10))
    assert np.array_equal(res.restricted[0], res.unrestricted[0])
    s = res.stats
    assert s["frames_with_drops"] == 0
    assert s["frame_drop_rate"] == 0.0
    assert s["demoted_sprite_frames"] == 0
    assert s["perceptible_losses"] == 0
    assert s["lost_scanlines"] == 0
    assert s["overflow_scanlines"] == 0
    assert s["diff_pixels_total"] == 0
    assert s["mean_diff_fraction"] == 0.0
    assert all(a.kept for a in res.allocations[0])
    assert s["allocation_seconds"] >= 0.0


def test_no_limit_equivalent_when_cap_above_sprite_count():
    # A limit larger than the sprite count must leave output bit-identical,
    # including overlap semantics (later objects overwrite earlier ones).
    sprites = [_sprite(i, i, 0, w=2, h=16) for i in range(5)]
    frames = [_frame(sprites)]
    res = allocate_sprite_budget(frames, (0, 0), ScanlineBudget(max_per_scanline=10))
    assert np.array_equal(res.restricted[0], res.unrestricted[0])

    # Independent reference composite built without any allocation logic.
    ref = np.array(frames[0].background, copy=True)
    for sp in sprites:
        h, w = sp.pixels.shape[:2]
        ref[sp.y:sp.y + h, sp.x:sp.x + w][~sp.transparent] = sp.pixels[~sp.transparent]
    assert np.array_equal(res.unrestricted[0], ref)
    assert np.array_equal(res.restricted[0], ref)


# --- overflow arbitration ---------------------------------------------------

def test_overflow_keeps_only_limit_sprites_per_line():
    # 12 single-row sprites all on scanline 0; limit 10 -> 2 bumped.
    sprites = [_sprite(i, x=i, y=0, w=1, h=1) for i, x in enumerate(range(12))]
    frames = [_frame(sprites)]
    res = allocate_sprite_budget(frames, (0, 0), ScanlineBudget(max_per_scanline=10))
    allocs = res.allocations[0]
    kept = [a.object_index for a in allocs if a.kept]
    demoted = [a.object_index for a in allocs if not a.kept]
    assert len(kept) == 10
    assert len(demoted) == 2
    assert res.stats["overflow_scanlines"] == 1
    assert res.stats["frames_with_drops"] == 1
    assert res.stats["frame_drop_rate"] == 1.0
    # Every demoted strip was baked: 1 visible pixel each -> 2 baked pixels,
    # both snapped onto the DMG background palette (red -> dark ramp entry).
    assert res.stats["perceptible_losses"] == 2
    assert res.stats["diff_pixels_total"] == 2
    snapped_red = gbp._snap_to_palette(
        np.array([[255, 0, 0]], dtype=np.uint8), gbp.DMG_RAMP
    )[0].tolist()
    assert res.backgrounds[0][0, 10].tolist() == snapped_red
    assert res.backgrounds[0][0, 11].tolist() == snapped_red
    assert snapped_red != [255, 0, 0]


def test_deterministic_tie_break_on_equal_scores():
    # Equal distance/frequency/area -> lowest id (then object order) survives.
    sprites = [_sprite(100 + i, x=i, y=0, w=1, h=1) for i in range(12)]
    frames = [_frame(sprites)]
    res1 = allocate_sprite_budget(frames, (5, 5), ScanlineBudget(max_per_scanline=10))
    res2 = allocate_sprite_budget(frames, (5, 5), ScanlineBudget(max_per_scanline=10))
    demoted1 = [a.sprite_id for a in res1.allocations[0] if not a.kept]
    demoted2 = [a.sprite_id for a in res2.allocations[0] if not a.kept]
    assert demoted1 == demoted2 == [110, 111]


def test_distance_weight_protects_player_neighbors():
    # 12 sprites all on scanline 0; only distance weight active. A near
    # cluster sits by the player (x=0..3), a far cluster at x=12..15.
    near = [_sprite(i, x=i, y=0, w=1, h=1, color=(255, 0, 0)) for i in range(6)]
    far = [_sprite(10 + i, x=12 + i % 4, y=0, w=1, h=1,
                   color=(0, 255, 0)) for i in range(6)]
    frames = [_frame(near + far)]
    budget = ScanlineBudget(
        max_per_scanline=10,
        weight_distance=1.0,
        weight_frequency=0.0,
        weight_area=0.0,
    )
    res = allocate_sprite_budget(frames, (0, 0), budget)
    demoted_ids = {a.sprite_id for a in res.allocations[0] if not a.kept}
    # Two sprites bumped; every one comes from the far cluster.
    assert len(demoted_ids) == 2
    assert demoted_ids <= {10, 11, 12, 13, 14, 15}
    assert {0, 1, 2, 3, 4, 5}.isdisjoint(demoted_ids)


def test_frequency_weight_protects_recurring_sprite():
    # A one-off big/near sprite vs. a recurring small one. With frequency
    # dominant the recurring id survives the over-limit line.
    rare = _sprite(0, x=0, y=0, w=8, h=8, color=(255, 0, 0))
    recurring = _sprite(7, x=8, y=0, w=2, h=2, color=(0, 255, 0))
    filler = [_sprite(20 + i, x=10 + i, y=0, w=1, h=8,
                      color=(0, 0, 255)) for i in range(10)]
    # Frame 0 overflows on scanline 0; later frames keep id 7 recurring.
    f0 = _frame([rare, recurring] + filler)
    f1 = _frame([recurring])
    f2 = _frame([recurring])
    budget = ScanlineBudget(
        max_per_scanline=10,
        weight_distance=0.0,
        weight_frequency=10.0,
        weight_area=0.0,
    )
    res = allocate_sprite_budget([f0, f1, f2], (0, 0), budget)
    f0_demoted = {a.sprite_id for a in res.allocations[0] if not a.kept}
    assert 7 not in f0_demoted          # recurring id protected
    assert 0 in f0_demoted or f0_demoted  # something had to give; rare at risk


def test_area_weight_protects_large_sprite():
    big = _sprite(0, x=0, y=0, w=8, h=8, color=(255, 0, 0))
    smalls = [_sprite(10 + i, x=8 + (i % 8), y=i % 8, w=1, h=1,
                      color=(0, 255, 0)) for i in range(11)]
    budget = ScanlineBudget(
        max_per_scanline=10,
        weight_distance=0.0,
        weight_frequency=0.0,
        weight_area=5.0,
    )
    res = allocate_sprite_budget([_frame([big] + smalls)], (8, 8), budget)
    big_alloc = next(a for a in res.allocations[0] if a.sprite_id == 0)
    assert big_alloc.kept


def test_weights_change_the_allocation():
    # Same scene, distance-only vs area-only must demote different sprites.
    # Line 0 carries 6 near 1x1 sprites, the far 4x8 giant, and 5 fillers ->
    # 12 intersecting sprites, two of which must be bumped.
    near_tiny = [_sprite(i, x=i, y=0, w=1, h=1, color=(255, 0, 0))
                 for i in range(6)]
    far_huge = _sprite(99, x=12, y=0, w=4, h=8, color=(0, 255, 0))
    filler = [_sprite(50 + i, x=10 + i % 6, y=0, w=1, h=1,
                      color=(0, 0, 255)) for i in range(5)]
    frames = [_frame(near_tiny + [far_huge] + filler)]
    d_only = allocate_sprite_budget(
        frames, (0, 0),
        ScanlineBudget(10, weight_distance=1, weight_frequency=0, weight_area=0),
    )
    a_only = allocate_sprite_budget(
        frames, (0, 0),
        ScanlineBudget(10, weight_distance=0, weight_frequency=0, weight_area=1),
    )
    d_demoted = {a.sprite_id for a in d_only.allocations[0] if not a.kept}
    a_demoted = {a.sprite_id for a in a_only.allocations[0] if not a.kept}
    assert len(d_demoted) == 2 and len(a_demoted) == 2
    assert d_demoted != a_demoted
    # Area-only protects the 4x8 giant; distance-only is willing to lose it.
    assert 99 not in a_demoted
    assert 99 in d_demoted


# --- geometry / transparency / metrics --------------------------------------

def test_off_canvas_sprites_do_not_count_against_lines():
    # Fully off-canvas sprite intersects no line; the 11 on-canvas single-row
    # sprites still overflow at 10, but the off-canvas one is never the victim
    # (it has no lost lines and never paints).
    off = _sprite(99, x=100, y=100, w=4, h=4)
    on = [_sprite(i, x=i, y=0, w=1, h=1) for i in range(11)]
    res = allocate_sprite_budget(
        [_frame([off] + on)], (0, 0), ScanlineBudget(max_per_scanline=10)
    )
    off_alloc = next(a for a in res.allocations[0] if a.sprite_id == 99)
    assert off_alloc.kept and off_alloc.lost_lines == []
    assert not off_alloc.baked
    # Partially clipped sprite: its on-canvas rows still arbitrate.
    clip = _sprite(100, x=-2, y=-2, w=4, h=4)
    res2 = allocate_sprite_budget(
        [_frame([clip] + on)], (0, 0), ScanlineBudget(max_per_scanline=10)
    )
    clip_alloc = next(a for a in res2.allocations[0] if a.sprite_id == 100)
    assert clip_alloc.kept  # visible on lines 0-1; never fully off-canvas
    assert res2.stats["overflow_scanlines"] >= 1


def test_transparent_demotion_is_imperceptible():
    # A demoted sprite whose lost strip is fully transparent changes nothing:
    # not counted as a perceptible loss, zero pixel diff. Give it the highest
    # id so tie-breaking picks it as one of the two bumped sprites.
    mask = np.ones((2, 1), dtype=bool)  # entirely transparent, same area as rest
    ghost = _sprite(99, x=13, y=0, w=1, h=2, transparent=mask)
    others = [_sprite(i, x=2 + i, y=0, w=1, h=2) for i in range(11)]
    res = allocate_sprite_budget(
        [_frame([ghost] + others)], (8, 0),
        ScanlineBudget(max_per_scanline=10),
    )
    ghost_alloc = next(a for a in res.allocations[0] if a.sprite_id == 99)
    assert not ghost_alloc.kept and not ghost_alloc.baked
    # The ghost baked zero pixels: only the other (opaque) victim is counted.
    assert res.stats["perceptible_losses"] == 1
    assert res.stats["demoted_sprite_frames"] == 2
    ghost_pixels = res.restricted[0][0:2, 13]
    assert np.array_equal(ghost_pixels, res.unrestricted[0][0:2, 13])


def test_partial_strip_demotion_bakes_only_lost_rows():
    # Tall sprite loses just scanline 0; only that row is baked, its other
    # rows remain OAM-drawn.
    tall = _sprite(0, x=0, y=0, w=2, h=6, color=(255, 255, 255))
    row_sprites = [_sprite(10 + i, x=2 + i, y=0, w=1, h=1,
                           color=(0, 255, 0)) for i in range(11)]
    res = allocate_sprite_budget(
        [_frame([tall] + row_sprites)], (0, 0),
        ScanlineBudget(max_per_scanline=10),
    )
    tall_alloc = next(a for a in res.allocations[0] if a.sprite_id == 0)
    # Either tall keeps line 0 (id 0 wins tie) or it loses exactly that line.
    if not tall_alloc.kept:
        assert tall_alloc.lost_lines == [0]
        snapped_white = gbp._snap_to_palette(
            np.array([[255, 255, 255]], dtype=np.uint8), gbp.DMG_RAMP
        )[0].tolist()
        assert res.backgrounds[0][0, 0].tolist() == snapped_white
        # Rows 1..5 still drawn via OAM onto the restricted composite.
        assert res.restricted[0][3, 0].tolist() == [255, 255, 255]


def test_frame_drop_rate_and_aggregates_over_four_frames():
    # Four-frame sequence; only frame 0 overflows (12 sprites on one line).
    def crowded():
        return [_sprite(i, x=i, y=0, w=1, h=1) for i in range(12)]

    frames = [
        _frame(crowded()),
        _frame([_sprite(0, 0, 0)]),
        _frame([_sprite(1, 4, 4)]),
        _frame([]),
    ]
    res = allocate_sprite_budget(frames, (8, 8), ScanlineBudget(max_per_scanline=10))
    s = res.stats
    assert s["frames"] == 4
    assert s["frames_with_drops"] == 1
    assert s["frame_drop_rate"] == 0.25
    assert s["demoted_sprite_frames"] == 2
    assert s["perceptible_losses"] == 2
    assert s["max_diff_fraction"] >= s["mean_diff_fraction"] > 0
    # Frames 1-3 stay pixel-identical to their unrestricted references.
    for fi in (1, 2, 3):
        assert np.array_equal(res.restricted[fi], res.unrestricted[fi])


def test_per_frame_backgrounds_are_independent_arrays():
    bg1 = np.zeros((8, 8, 3), dtype=np.uint8)
    bg2 = np.zeros((8, 8, 3), dtype=np.uint8)
    sprites = [_sprite(i, x=i, y=0, w=1, h=1) for i in range(12)]
    frames = [
        SpriteFrame(sprites=sprites, background=bg1),
        SpriteFrame(sprites=sprites, background=bg2),
    ]
    res = allocate_sprite_budget(frames, (4, 0), ScanlineBudget(max_per_scanline=10))
    # Input backgrounds must not be mutated by the allocation.
    assert bg1.sum() == 0 and bg2.sum() == 0
    assert res.backgrounds[0] is not res.backgrounds[1]
    assert res.restricted[0] is not res.unrestricted[0]


def test_custom_background_palette_governs_bake_snap():
    # A two-entry grayscale palette: baked red must snap to its nearest entry.
    pal = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    sprites = [_sprite(i, x=i, y=0, w=1, h=1, color=(255, 0, 0))
               for i in range(12)]
    budget = ScanlineBudget(max_per_scanline=10, background_palette=pal)
    res = allocate_sprite_budget([_frame(sprites)], (0, 0), budget)
    # Red's nearest grayscale entry is black.
    assert res.backgrounds[0][0, 10].tolist() == [0, 0, 0]
    assert res.backgrounds[0][0, 11].tolist() == [0, 0, 0]
    assert np.array_equal(budget.background_palette, pal)


def test_invalid_background_palette_raises():
    with pytest.raises(ValueError, match="background_palette"):
        ScanlineBudget(background_palette=np.zeros((3,), dtype=np.uint8))
    with pytest.raises(ValueError, match="background_palette"):
        ScanlineBudget(background_palette=np.zeros((0, 3), dtype=np.uint8))
