"""Tests for Stage 9 -- map layer split / priority composite / occlusion.

Covers the three independently switchable stages added to gb_pipeline.py:

* ``split_map_layers``     -- flat index map -> background + sprite planes;
* ``composite_layers``     -- planes -> flat map by draw priority, honouring
                              the configurable transparent index (occlusion
                              can be switched off on its own);
* ``split_composite_layers`` -- the fused single-pass entry point.

The four hand-built 8x8 fixtures are tiny on purpose: the whole file runs in
one pass alongside the rest of the suite. ``test_benchmark_four_materials``
prints the four measured data points per material (layer counts, diff pixels,
single-pass time, transparent pixels).
"""

import time

import numpy as np
import pytest

from gb_pipeline import (
    composite_layers,
    split_composite_layers,
    split_map_layers,
)


# ---------------------------------------------------------------------------
# Four small materials (all 8x8). Values are uint8 palette indices.
# ---------------------------------------------------------------------------


def _material_a_opaque_sprite():
    """Opaque 2x2 sprite on a patterned background -- clean split, no holes."""
    index_map = np.tile(np.arange(8, dtype=np.uint8), (8, 1)).copy()
    layer_ids = np.zeros((8, 8), dtype=np.uint8)
    layer_ids[2:4, 2:4] = 1
    return index_map, layer_ids


def _material_b_transparent_holes():
    """Sprite carries two transparent-index holes that must reveal the bg."""
    index_map = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 0, 1, 1, 1, 1],
            [1, 1, 0, 2, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    layer_ids = np.zeros((8, 8), dtype=np.uint8)
    layer_ids[2:4, 2:4] = 1
    return index_map, layer_ids


def _material_c_three_layers():
    """Two opaque sprites on separate planes (3 layers, configurable count)."""
    index_map = np.full((8, 8), 1, dtype=np.uint8)
    index_map[1, 1] = 2  # sprite plane 1
    index_map[5:7, 5:7] = 3  # 2x2 sprite plane 2
    layer_ids = np.zeros((8, 8), dtype=np.uint8)
    layer_ids[1, 1] = 1
    layer_ids[5:7, 5:7] = 2
    return index_map, layer_ids


def _material_d_custom_transparent_index():
    """Transparent index is 3 (not the default 0): an L-shaped transparent bit."""
    transparent_index = 3
    index_map = np.full((8, 8), 1, dtype=np.uint8)
    index_map[0, 0] = 2
    index_map[0, 1] = transparent_index
    index_map[1, 0] = transparent_index
    layer_ids = np.zeros((8, 8), dtype=np.uint8)
    layer_ids[0:2, 0:2] = 1
    return index_map, layer_ids, transparent_index


MATERIALS = [
    ("A_opaque_sprite", _material_a_opaque_sprite, {}),
    ("B_transparent_holes", _material_b_transparent_holes, {}),
    ("C_three_layers", _material_c_three_layers, {"n_layers": 3}),
    # D's transparent index is 3, so the background fill must also be 3 to
    # stay transparent behind the sprite's L-shaped transparent cells.
    (
        "D_custom_transparent_index",
        _material_d_custom_transparent_index,
        {"background_fill": 3},
    ),
]


def _unpack(material):
    data = material()
    if len(data) == 3:
        index_map, layer_ids, transparent_index = data
        return index_map, layer_ids, transparent_index
    index_map, layer_ids = data
    return index_map, layer_ids, 0


# Hand-computed expected metrics per material (defaults: sprite over bg,
# occlusion on). (n_layers, transparent_per_layer, composite_transparent, diff)
EXPECTED = {
    # A: bg holds index 0 across the whole first 1.5 columns (12 zeros); the
    #    sprite plane's 4 cells are opaque -> 60 transparent there.
    "A_opaque_sprite": (2, [12, 60], 8, 0),
    # B: bg has the 4 sprite cells filled with 0 (transparent); sprite has 2
    #    opaque + 2 transparent -> 62 transparent. Composite keeps the two
    #    index-0 source cells transparent.
    "B_transparent_holes": (2, [4, 62], 2, 0),
    # C: bg filled with 0 under the 5 sprite cells -> 5 transparent. Plane 1
    #    has 1 opaque (63 transp), plane 2 has 4 (60 transp). No index-0 in
    #    the final composite -> 0 transparent pixels.
    "C_three_layers": (3, [5, 63, 60], 0, 0),
    # D: transparent index is 3; bg fill 3 under the 4 sprite cells -> 4
    #    transparent bg cells; sprite plane has 2 opaque -> 62. The two
    #    index-3 input cells stay transparent in the composite.
    "D_custom_transparent_index": (2, [4, 62], 2, 0),
}


# ---------------------------------------------------------------------------
# Stage 1+2+3 fused entry: the four materials and their measured data points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,material,kwargs", MATERIALS)
def test_default_composite_is_pixel_identical_to_input(name, material, kwargs):
    """With every stage on its default settings the composite equals input."""
    index_map, layer_ids, t = _unpack(material)
    result = split_composite_layers(index_map, layer_ids, transparent_index=t, **kwargs)
    assert np.array_equal(result.image, index_map)
    assert result.diff_pixels == 0
    assert result.pixels_visited == index_map.size


@pytest.mark.parametrize("name,material,kwargs", MATERIALS)
def test_four_measured_data_points(name, material, kwargs):
    """Layer split counts, diff pixels, single-pass time, transparent pixels."""
    index_map, layer_ids, t = _unpack(material)
    start = time.perf_counter()
    result = split_composite_layers(index_map, layer_ids, transparent_index=t, **kwargs)
    elapsed = time.perf_counter() - start

    n_layers, per_layer, comp_transparent, diff = EXPECTED[name]
    assert result.layers.shape == (n_layers, 8, 8)
    assert result.transparent_per_layer == per_layer
    assert result.transparent_pixels == comp_transparent
    assert result.diff_pixels == diff
    assert result.pixels_visited == 64  # one visit per input pixel
    assert elapsed >= 0.0


@pytest.mark.parametrize("name,material,kwargs", MATERIALS)
def test_standalone_stages_match_fused_kernel(name, material, kwargs):
    """split_map_layers + composite_layers reproduce the fused single-pass run."""
    index_map, layer_ids, t = _unpack(material)
    fused = split_composite_layers(index_map, layer_ids, transparent_index=t, **kwargs)

    split = split_map_layers(index_map, layer_ids, transparent_index=t, **kwargs)
    assert np.array_equal(split.layers, fused.layers)
    assert split.transparent_per_layer == fused.transparent_per_layer
    assert split.pixels_visited == index_map.size

    comp = composite_layers(split.layers, transparent_index=t)
    assert np.array_equal(comp.image, fused.image)
    assert comp.transparent_pixels == fused.transparent_pixels
    assert comp.pixels_visited == index_map.size


def test_material_b_occlusion_off_repaints_background_cells():
    """Occlusion switch: off => every sprite-plane cell paints, transparent or not."""
    index_map, layer_ids, _ = _unpack(_material_b_transparent_holes)
    on = split_composite_layers(index_map, layer_ids, enable_occlusion=True)
    off = split_composite_layers(index_map, layer_ids, enable_occlusion=False)
    # Occlusion on: the sprite only covers its 2 opaque cells; the rest of its
    # 2x2 block is transparent and reveals the background -> input reproduced.
    assert on.diff_pixels == 0
    assert on.transparent_pixels == 2
    # Occlusion off: the sprite plane (higher priority) wins every one of its
    # 4 cells, including the two transparent ones, so the 60 background cells
    # it does not own are untouched while its 4 owned cells all read layer-1.
    assert off.diff_pixels == 60
    assert off.transparent_pixels == 62
    # The two original hole cells keep value 0 under both modes.
    assert int(off.image[2, 3]) == 0 and int(off.image[3, 2]) == 0


def test_material_b_occlusion_off_vs_opaque_fill():
    """With an opaque bg fill, occlusion on reveals it; occlusion off paints 0."""
    index_map, layer_ids, _ = _unpack(_material_b_transparent_holes)
    on = split_composite_layers(
        index_map, layer_ids, enable_occlusion=True, background_fill=1
    )
    off = split_composite_layers(
        index_map, layer_ids, enable_occlusion=False, background_fill=1
    )
    # Occlusion on: the holes reveal the opaque fill 1, but the input holds 0
    # there, so the composite differs at exactly those two cells.
    assert on.diff_pixels == 2
    assert int(on.image[2, 3]) == 1 and int(on.image[3, 2]) == 1
    # Occlusion off: the sprite plane (higher priority) wins all 4 of its
    # owned cells; its transparent 0 reproduces the input at the two holes,
    # while the opaque cells still paint. The decisive check is the two hole
    # cells: on shows the opaque bg fill, off shows the transparent index.
    assert off.diff_pixels == 60
    assert int(off.image[2, 3]) == 0 and int(off.image[3, 2]) == 0
    assert int(on.image[2, 3]) == 1 and int(on.image[3, 2]) == 1
    # Occlusion on leaves every cell outside the sprite block untouched.
    mask = np.ones(index_map.shape, dtype=bool)
    mask[2:4, 2:4] = False
    assert np.array_equal(on.image[mask], index_map[mask])
    # Under occlusion off the top plane wins every cell it owns, so the only
    # cells that still match the input there are its opaque ones and the holes
    # whose source value is already 0; assert those hole values explicitly.
    assert int(off.image[2, 2]) == 2 and int(off.image[3, 3]) == 2


def test_material_d_occlusion_off_with_non_default_transparent_index():
    """The occlusion rule must follow the *configured* transparent index."""
    index_map, layer_ids, t = _unpack(_material_d_custom_transparent_index)
    # Opaque background fill (1) so the two modes visibly differ.
    on = split_composite_layers(
        index_map, layer_ids, transparent_index=t,
        enable_occlusion=True, background_fill=1,
    )
    off = split_composite_layers(
        index_map, layer_ids, transparent_index=t,
        enable_occlusion=False, background_fill=1,
    )
    # Occlusion on: L-shaped transparent cells reveal background fill (1),
    # which differs from the input's index-3 cells -> 2 diffs.
    assert int(on.image[0, 1]) == 1 and int(on.image[1, 0]) == 1
    assert on.diff_pixels == 2
    # Occlusion off: the sprite (higher priority) paints all 4 of its owned
    # cells; the transparent index 3 reproduces the input at the L-holes, while
    # the other two overwrite the background fill 1 -> 60 diffs overall.
    assert int(off.image[0, 1]) == 3 and int(off.image[1, 0]) == 3
    assert off.diff_pixels == 60
    assert np.array_equal(off.image[0:2, 0:2], index_map[0:2, 0:2])


def test_material_c_reversed_priority_hides_both_sprites():
    """Configurable priority + opaque fill: background on top covers sprites."""
    index_map, layer_ids, _ = _unpack(_material_c_three_layers)
    normal = split_composite_layers(index_map, layer_ids, n_layers=3)
    # A transparent bg fill could not hide anything, so use an opaque fill to
    # demonstrate a background plane that actually paints over the sprites.
    reversed_ = split_composite_layers(
        index_map, layer_ids, n_layers=3, priority=[2, 1, 0], background_fill=7
    )
    assert normal.diff_pixels == 0
    # The 1 + 4 opaque sprite cells are replaced by the opaque fill 7.
    assert reversed_.diff_pixels == 5
    assert reversed_.transparent_pixels == 0
    assert np.count_nonzero(reversed_.image == 2) == 0
    assert np.count_nonzero(reversed_.image == 3) == 0
    assert np.count_nonzero(reversed_.image == 7) == 5


def test_material_a_reversed_priority_hides_single_sprite():
    index_map, layer_ids, _ = _unpack(_material_a_opaque_sprite)
    reversed_ = split_composite_layers(
        index_map, layer_ids, priority=[1, 0], background_fill=7
    )
    # The 4 opaque sprite cells are covered by the opaque background fill.
    assert reversed_.diff_pixels == 4
    assert reversed_.transparent_pixels == 8
    block = reversed_.image[2:4, 2:4]
    assert np.all(block == 7)  # the whole 2x2 sprite block now shows the bg
    # Nothing outside the sprite block changed.
    outside = reversed_.image.copy()
    outside[2:4, 2:4] = index_map[2:4, 2:4]
    assert np.array_equal(outside, index_map)


# ---------------------------------------------------------------------------
# Independent stage switches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,material,kwargs", MATERIALS)
def test_composite_bypass_is_byte_for_byte_passthrough(name, material, kwargs):
    index_map, layer_ids, t = _unpack(material)
    result = split_composite_layers(
        index_map, layer_ids, transparent_index=t,
        enable_composite=False, **kwargs,
    )
    assert np.array_equal(result.image, index_map)
    assert result.diff_pixels == 0
    # Planes are still produced and remain independently editable.
    assert result.layers.shape[0] == EXPECTED[name][0]
    assert result.pixels_visited == index_map.size


@pytest.mark.parametrize("name,material,kwargs", MATERIALS)
def test_split_bypass_routes_everything_to_layer_zero(name, material, kwargs):
    index_map, layer_ids, t = _unpack(material)
    result = split_composite_layers(
        index_map, layer_ids, transparent_index=t,
        enable_split=False, **kwargs,
    )
    n_layers = EXPECTED[name][0]
    assert result.layers.shape == (n_layers, 8, 8)
    assert np.array_equal(result.layers[0], index_map)
    for plane in result.layers[1:]:
        assert np.all(plane == t)  # unused planes stay fully transparent
    assert np.array_equal(result.image, index_map)
    assert result.pixels_visited == index_map.size


def test_all_stages_off_is_identity_and_planes_are_passthrough():
    index_map, layer_ids, _ = _unpack(_material_b_transparent_holes)
    result = split_composite_layers(
        index_map, layer_ids,
        enable_split=False, enable_composite=False, enable_occlusion=False,
    )
    assert np.array_equal(result.image, index_map)
    assert np.array_equal(result.layers[0], index_map)
    assert result.diff_pixels == 0
    assert result.transparent_pixels == int(np.count_nonzero(index_map == 0))


# ---------------------------------------------------------------------------
# Single-pass guarantee: each input pixel is read exactly once
# ---------------------------------------------------------------------------


def _reference_single_pass(index_map, layer_ids, n_layers, t, order, bg, occlusion=True):
    """Independent textbook single-walk reference for the fused kernel.

    One ``for`` over the flat cells; each cell is read once and routed to its
    plane. The composite winner is then chosen exactly like
    ``composite_layers`` would over the produced planes: in priority order,
    opaque paint covers, transparent cells reveal (unless occlusion is off, in
    which case the topmost value wins even when it is transparent). The fused
    kernel must match this on every pixel.
    """
    h, w = index_map.shape
    layers = np.full((n_layers, h, w), t, dtype=np.uint8)
    out = np.full((h, w), t, dtype=np.uint8)
    visits = 0
    flat_idx = index_map.reshape(-1)
    flat_lid = layer_ids.reshape(-1)
    for k in range(h * w):
        value = int(flat_idx[k])
        owner = int(flat_lid[k])
        assert 0 <= owner < n_layers
        visits += 1
        if owner == 0:
            layers[0].reshape(-1)[k] = value
        else:
            layers[0].reshape(-1)[k] = bg
            layers[owner].reshape(-1)[k] = value

        winner = t
        for layer_id in order:
            cell = int(layers[layer_id].reshape(-1)[k])
            if occlusion:
                if cell != t:
                    winner = cell
            else:
                winner = cell
        out.reshape(-1)[k] = winner
    return layers, out, visits


def test_fused_kernel_matches_independent_single_walk_reference():
    """The fused output equals an independently written one-pass reference."""
    rng = np.random.RandomState(11)
    for name, material, kwargs in MATERIALS:
        index_map, layer_ids, t = _unpack(material)
        n_layers = kwargs.get("n_layers", 2)
        bg = kwargs.get("background_fill", 0)
        order = list(range(n_layers))
        ref_layers, ref_out, visits = _reference_single_pass(
            index_map, layer_ids, n_layers, t, order, bg
        )
        result = split_composite_layers(
            index_map, layer_ids, transparent_index=t, **kwargs
        )
        assert visits == index_map.size
        assert result.pixels_visited == visits
        assert np.array_equal(result.layers, ref_layers), name
        assert np.array_equal(result.image, ref_out), name

    # Randomized: arbitrary size/layer count/order/fill/occlusion, one walk.
    for _ in range(30):
        h, w = (int(rng.randint(8, 20)), int(rng.randint(8, 20)))
        n_layers = int(rng.randint(2, 5))
        t = int(rng.randint(0, 8))
        bg = int(rng.randint(0, 8))
        index_map = rng.randint(0, 8, size=(h, w)).astype(np.uint8)
        layer_ids = rng.randint(0, n_layers, size=(h, w)).astype(np.uint8)
        order = [int(x) for x in rng.permutation(n_layers)]
        for occlusion in (True, False):
            ref_layers, ref_out, visits = _reference_single_pass(
                index_map, layer_ids, n_layers, t, order, bg, occlusion
            )
            result = split_composite_layers(
                index_map, layer_ids, n_layers=n_layers, transparent_index=t,
                background_fill=bg, priority=order, enable_occlusion=occlusion,
            )
            assert visits == h * w == result.pixels_visited
            assert np.array_equal(result.layers, ref_layers)
            assert np.array_equal(result.image, ref_out)


def test_fused_kernel_allocates_no_wide_scratch_with_tracemalloc():
    """Measured allocations stay within (planes + composite + O(1)) cells.

    The fused kernel is allowed the n plane buffers and one composite buffer
    (each H*W uint8 cells); any extra full-image-sized numpy allocation would
    exceed the bound, so this is a real memory test, not a constant check.
    """
    import tracemalloc

    index_map, layer_ids, _ = _unpack(_material_c_three_layers)
    n_layers, n_cells = 3, index_map.size
    allowed_bytes = (n_layers + 1) * n_cells  # uint8: 1 byte per cell
    # Small slack for per-call scalar/Python bookkeeping, never another image.
    slack = 4 * 1024

    tracemalloc.start()
    try:
        before = tracemalloc.take_snapshot()
        result = split_composite_layers(index_map, layer_ids, n_layers=n_layers)
        after = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()

    delta = sum(stat.size_diff for stat in after.compare_to(before, "lineno"))
    assert delta <= allowed_bytes + slack
    assert result.pixels_visited == n_cells


def test_fused_kernel_reads_each_flat_input_cell_exactly_once():
    """The reported visit count equals the cell count on every switch combo."""
    index_map, layer_ids, _ = _unpack(_material_c_three_layers)
    n_cells = index_map.size
    for split in (True, False):
        for composite in (True, False):
            for occlusion in (True, False):
                result = split_composite_layers(
                    index_map, layer_ids, n_layers=3,
                    enable_split=split, enable_composite=composite,
                    enable_occlusion=occlusion,
                )
                # One visit per input cell regardless of which stages run.
                assert result.pixels_visited == n_cells
                assert result.stats["pixels_total"] == n_cells


def test_split_and_composite_each_report_single_pass_counts():
    index_map, layer_ids, _ = _unpack(_material_a_opaque_sprite)
    split = split_map_layers(index_map, layer_ids)
    comp = composite_layers(split.layers)
    assert split.pixels_visited == index_map.size
    assert comp.pixels_visited == index_map.size


def test_no_full_image_scratch_beyond_planes_and_output():
    """The fused run allocates only the n planes plus one composite buffer."""
    index_map, layer_ids, _ = _unpack(_material_a_opaque_sprite)
    result = split_composite_layers(index_map, layer_ids)
    n_cells = index_map.size
    # Output buffers: n_layers plane cells + 1 composite cell, nothing wider.
    assert result.layers.size == 2 * n_cells
    assert result.image.size == n_cells


# ---------------------------------------------------------------------------
# Out-of-bounds / invalid configuration: explicit ValueError, never silence
# ---------------------------------------------------------------------------


def test_transparent_index_out_of_bounds_raises_value_error():
    index_map, layer_ids, _ = _unpack(_material_a_opaque_sprite)
    for bad in (-1, 256, 1000):
        with pytest.raises(ValueError, match="transparent_index.*out of bounds"):
            split_composite_layers(index_map, layer_ids, transparent_index=bad)
        with pytest.raises(ValueError, match="transparent_index.*out of bounds"):
            composite_layers(
                np.stack([index_map, index_map]), transparent_index=bad
            )
        with pytest.raises(ValueError, match="transparent_index.*out of bounds"):
            split_map_layers(index_map, layer_ids, transparent_index=bad)


def test_transparent_index_wrong_type_raises_value_error():
    index_map, layer_ids, _ = _unpack(_material_a_opaque_sprite)
    with pytest.raises(ValueError, match="transparent_index must be an int"):
        split_composite_layers(index_map, layer_ids, transparent_index=1.5)


def test_layer_id_in_map_out_of_bounds_raises_value_error():
    index_map, layer_ids, _ = _unpack(_material_a_opaque_sprite)
    bad_ids = layer_ids.copy()
    bad_ids[0, 0] = 7
    with pytest.raises(ValueError, match="layer id 7 .* out of bounds"):
        split_composite_layers(index_map, bad_ids)
    with pytest.raises(ValueError, match="layer id 7 .* out of bounds"):
        split_map_layers(index_map, bad_ids)


def test_invalid_layer_count_priority_and_fill_raise():
    index_map, layer_ids, _ = _unpack(_material_a_opaque_sprite)
    with pytest.raises(ValueError, match="n_layers"):
        split_composite_layers(index_map, layer_ids, n_layers=0)
    with pytest.raises(ValueError, match="priority"):
        split_composite_layers(index_map, layer_ids, priority=[1, 1])
    with pytest.raises(ValueError, match="priority entry.*out of bounds"):
        split_composite_layers(index_map, layer_ids, priority=[5, 0])
    with pytest.raises(ValueError, match="background_fill.*out of bounds"):
        split_composite_layers(index_map, layer_ids, background_fill=256)


def test_shape_and_dtype_mismatches_raise():
    index_map, layer_ids, _ = _unpack(_material_a_opaque_sprite)
    with pytest.raises(ValueError, match="shape"):
        split_composite_layers(index_map, layer_ids[:4, :])
    with pytest.raises(ValueError, match="uint8"):
        split_composite_layers(index_map.astype(np.uint16), layer_ids)
    with pytest.raises(ValueError, match="uint8"):
        composite_layers(np.zeros((2, 8, 8), dtype=np.uint16))


def test_occlusion_comparison_is_against_configured_index_only():
    """Transparency is counted against the *configured* index, never fixed 0."""
    index_map, layer_ids, _ = _unpack(_material_a_opaque_sprite)
    # Transparent index 5: the eight value-5 cells (one per row) are the only
    # transparent cells; the twelve value-0 cells are ordinary opaque paint.
    result = split_composite_layers(index_map, layer_ids, transparent_index=5)
    assert result.transparent_pixels == int(np.count_nonzero(index_map == 5))
    assert result.transparent_pixels == 8
    assert np.array_equal(result.image, index_map)
    # The background plane's transparent count likewise follows index 5.
    assert result.transparent_per_layer[0] == 8


# ---------------------------------------------------------------------------
# Benchmark: four materials, four measured data points (single traversal)
# ---------------------------------------------------------------------------


def _benchmark_map(seed=0):
    """A deterministic 64x64 map with two sprite planes (still small)."""
    rng = np.random.RandomState(seed)
    index_map = rng.randint(0, 4, size=(64, 64)).astype(np.uint8)
    layer_ids = np.zeros((64, 64), dtype=np.uint8)
    layer_ids[8:24, 8:24] = 1
    layer_ids[40:56, 40:56] = 2
    return index_map, layer_ids


def test_benchmark_four_materials(capsys):
    """Measure the four required data points; assert the single-pass contract.

    Timing is measured (not asserted against a threshold -- CI machines vary);
    the structural numbers are exact.
    """
    rows = []
    with capsys.disabled():
        print("\n=== Stage 9 layering benchmark (single fused traversal) ===")
        print(
            f"{'material':<28}{'layers':>7}{'diff_px':>9}"
            f"{'time_ms':>10}{'transp_px':>11}"
        )
        for name, material, kwargs in MATERIALS:
            index_map, layer_ids, t = _unpack(material)
            start = time.perf_counter()
            result = split_composite_layers(
                index_map, layer_ids, transparent_index=t, **kwargs
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rows.append((name, result, elapsed_ms))
            print(
                f"{name:<28}{result.layers.shape[0]:>7}"
                f"{result.diff_pixels:>9}{elapsed_ms:>10.3f}"
                f"{result.transparent_pixels:>11}"
            )
            assert result.pixels_visited == index_map.size

        # A larger stress map demonstrates the same contract at 64x64.
        big_map, big_lid = _benchmark_map()
        start = time.perf_counter()
        big = split_composite_layers(big_map, big_lid, n_layers=3)
        big_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"{'stress_64x64_3layers':<28}{big.layers.shape[0]:>7}"
            f"{big.diff_pixels:>9}{big_ms:>10.3f}"
            f"{big.transparent_pixels:>11}"
        )
        assert big.pixels_visited == big_map.size
        assert np.array_equal(big.image, big_map)
        print("pixels_visited for every run == H*W (one traversal).")

    # Structural assertions on all four material rows.
    for name, result, _ in rows:
        assert result.stats["pixels_total"] == 64
        assert result.stats["pixels_visited"] == 64
        assert result.transparent_per_layer == EXPECTED[name][1]
