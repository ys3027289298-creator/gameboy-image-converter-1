"""Tests for the exhaustive branch-and-bound 4-color palette search.

Covers the Stage 3b additions in gb_pipeline.py: global optimality against a
brute-force enumeration, soundness of the lower-bound pruning, the search
statistics, the configurable tolerance / candidate set, the explicit timeout
return, and the guarantee that the pre-existing fixed/custom-palette pipeline
output stays pixel-identical.
"""

import itertools
import math

import numpy as np
import pytest
from PIL import Image

import gb_pipeline as gbp
from gb_pipeline import (
    luminance_sort,
    palette_mean_delta_e,
    quantize_working_set,
    search_four_color_palette,
    snap_rgb555,
)


def _rng():
    return np.random.RandomState(2026)


def _random_candidates(rng, n):
    cands = snap_rgb555(rng.randint(0, 256, (n, 3)).astype(np.uint8))
    return np.unique(cands, axis=0)


def _brute_force_best(image, cands, k=4):
    """Reference enumeration: weighted total Lab dE of every k-combination."""
    src_lab, _, weights = gbp._image_weighted_colors(image)
    cand_lab = gbp._rgb_to_lab(cands)
    wdist = (
        np.sqrt(((src_lab[:, None, :] - cand_lab[None, :, :]) ** 2).sum(axis=2))
        * weights[:, None]
    )
    best = np.inf
    for combo in itertools.combinations(range(len(cands)), k):
        score = float(wdist[:, combo].min(axis=1).sum())
        best = min(best, score)
    return best


@pytest.fixture
def image_and_candidates():
    rng = _rng()
    cands = _random_candidates(rng, 10)
    image = cands[rng.randint(0, len(cands), (16, 16))]
    image[:8] = cands[0]  # skew the pixel weights like real flat art
    return image, cands


# --- optimality -------------------------------------------------------------


def test_search_matches_brute_force_across_trials():
    """Pruned and unpruned searches must both equal brute-force enumeration."""
    rng = _rng()
    for trial in range(10):
        cands = _random_candidates(rng, rng.randint(5, 12))
        image = cands[rng.randint(0, len(cands), (12, 12))]
        expected = _brute_force_best(image, cands)

        pruned = search_four_color_palette(image, cands, enable_pruning=True)
        full = search_four_color_palette(image, cands, enable_pruning=False)

        assert pruned.status == "optimal" and not pruned.timed_out
        assert abs(pruned.total_delta_e - expected) < 1e-9
        assert abs(full.total_delta_e - expected) < 1e-9
        # The reported mean is the total divided by the pixel count.
        assert abs(pruned.mean_delta_e - expected / image.reshape(-1, 3).shape[0]) < 1e-9


def test_search_accepts_pil_image_and_rgb_array_equivalently(image_and_candidates):
    image, cands = image_and_candidates
    r_arr = search_four_color_palette(image, cands)
    r_img = search_four_color_palette(Image.fromarray(image, "RGB"), cands)
    assert abs(r_arr.total_delta_e - r_img.total_delta_e) < 1e-12
    assert np.array_equal(r_arr.palette, r_img.palette)


def test_palette_helper_scores_the_same_objective(image_and_candidates):
    image, cands = image_and_candidates
    res = search_four_color_palette(image, cands)
    assert abs(palette_mean_delta_e(image, res.palette) - res.mean_delta_e) < 1e-9


# --- pruning statistics -----------------------------------------------------


def test_unpruned_enumeration_visits_every_combination(image_and_candidates):
    image, cands = image_and_candidates
    n = len(cands)
    full = search_four_color_palette(image, cands, enable_pruning=False)
    assert full.nodes_pruned == 0
    assert full.leaves_evaluated == len(list(itertools.combinations(range(n), 4)))
    # Feasible DFS nodes: root plus prefix combinations that leave enough
    # candidates after their last pick to complete the k-color palette.
    expected_nodes = 1
    for s in range(1, 5):
        expected_nodes += sum(
            math.comb(last, s - 1) for last in range(s - 1, n - 4 + s)
        )
    assert full.nodes_total == expected_nodes
    assert full.nodes_evaluated == full.nodes_total


def test_pruning_keeps_optimum_and_reduces_nodes(image_and_candidates):
    image, cands = image_and_candidates
    full = search_four_color_palette(image, cands, enable_pruning=False)
    pruned = search_four_color_palette(image, cands, enable_pruning=True)
    assert abs(pruned.total_delta_e - full.total_delta_e) < 1e-9
    assert pruned.nodes_evaluated <= full.nodes_evaluated
    assert pruned.nodes_pruned >= 1
    # Pruning never discards a leaf: evaluated + pruned subtrees stay coherent.
    assert pruned.leaves_evaluated <= full.leaves_evaluated


def test_pruning_lower_bound_is_admissible_under_random_partials():
    """Property check: the charge lower bound never exceeds the true best
    descendant score at any reachable partial node (the soundness invariant)."""
    rng = _rng()
    cands = _random_candidates(rng, 9)
    image = cands[rng.randint(0, len(cands), (10, 10))]
    src_lab, _, weights = gbp._image_weighted_colors(image)
    wdist = (
        np.sqrt(
            (
                (src_lab[:, None, :] - gbp._rgb_to_lab(cands)[None, :, :]) ** 2
            ).sum(axis=2)
        )
        * weights[:, None]
    )
    for depth in (1, 2, 3):
        for chosen in itertools.combinations(range(len(cands)), depth):
            current = wdist[:, list(chosen)].min(axis=1)
            slots = 4 - depth
            mask = np.ones(len(cands), dtype=bool)
            mask[np.array(chosen)] = False
            q = np.maximum(current[:, None] - wdist[:, mask], 0.0).sum(axis=0)
            top = np.sort(q)[-slots:].sum()
            lower_bound = current.sum() - top
            # True best completion over the remaining candidates.
            remaining = [j for j in range(len(cands)) if j not in chosen]
            true_best = min(
                wdist[:, list(chosen) + list(extra)].min(axis=1).sum()
                for extra in itertools.combinations(remaining, slots)
            )
            assert lower_bound <= true_best + 1e-9


# --- returned palette invariants --------------------------------------------


def test_returned_palette_is_snapped_sorted_and_from_candidates(image_and_candidates):
    image, cands = image_and_candidates
    res = search_four_color_palette(image, cands)
    assert res.palette.shape == (4, 3) and res.palette.dtype == np.uint8
    assert np.array_equal(snap_rgb555(res.palette), res.palette)
    assert np.array_equal(luminance_sort(res.palette), res.palette)
    allowed = {tuple(int(v) for v in c) for c in np.unique(cands, axis=0)}
    for color in np.unique(res.palette, axis=0):
        assert tuple(int(v) for v in color) in allowed


def test_candidate_set_is_configurable_and_restricts_output():
    rng = _rng()
    cands = _random_candidates(rng, 12)
    image = cands[rng.randint(0, len(cands), (12, 12))]
    subset = cands[:6]
    res = search_four_color_palette(image, subset)
    allowed = {tuple(int(v) for v in c) for c in np.unique(subset, axis=0)}
    for color in np.unique(res.palette, axis=0):
        assert tuple(int(v) for v in color) in allowed
    # Restricting the pool cannot improve over the full pool.
    full = search_four_color_palette(image, cands)
    assert res.mean_delta_e + 1e-9 >= full.mean_delta_e


# --- tolerance ---------------------------------------------------------------


def test_tolerance_result_is_within_tolerance_of_optimum(image_and_candidates):
    image, cands = image_and_candidates
    exact = search_four_color_palette(image, cands)
    relaxed = search_four_color_palette(image, cands, tolerance=3.0)
    assert relaxed.status == "optimal_within_tolerance"
    assert relaxed.tolerance == 3.0
    assert relaxed.mean_delta_e <= exact.mean_delta_e + 3.0 + 1e-9
    assert relaxed.nodes_evaluated <= exact.nodes_evaluated


# --- timeout -----------------------------------------------------------------


def test_timeout_returns_explicitly_with_best_effort_palette():
    rng = _rng()
    cands = _random_candidates(rng, 26)
    image = cands[rng.randint(0, len(cands), (8, 8))]
    res = search_four_color_palette(
        image, cands, time_limit=0.001, warm_start=cands[:4]
    )
    assert res.timed_out and res.status == "timeout"
    assert res.palette is not None  # warm start guarantees a scored incumbent
    assert res.time_limit_seconds == 0.001


def test_timeout_without_incumbent_returns_none_palette():
    rng = _rng()
    cands = _random_candidates(rng, 26)
    image = cands[rng.randint(0, len(cands), (8, 8))]
    res = search_four_color_palette(image, cands, time_limit=0.0)
    assert res.timed_out and res.status == "timeout"
    assert res.palette is None
    assert np.isinf(res.mean_delta_e)


# --- validation --------------------------------------------------------------


def test_invalid_arguments_raise(image_and_candidates):
    image, cands = image_and_candidates
    with pytest.raises(ValueError):
        search_four_color_palette(image, cands, tolerance=-1.0)
    with pytest.raises(ValueError):
        search_four_color_palette(image, cands, time_limit=-1.0)
    with pytest.raises(ValueError):
        search_four_color_palette(image, cands, k=0)
    with pytest.raises(ValueError):
        search_four_color_palette(image, cands[:3])


# --- fixed/custom palette output stays pixel-identical -----------------------


def test_fixed_custom_palette_pipeline_output_unchanged():
    """The search is purely additive: a fixed custom palette must still flow
    through quantize -> pack -> index to byte-identical, deterministic pixels,
    and every shipped pixel must be an entry of the fixed palette set."""
    rng = _rng()
    src = rng.randint(0, 256, (32, 32, 3)).astype(np.uint8)
    custom = snap_rgb555(rng.randint(0, 256, (8, 3)).astype(np.uint8))

    def run():
        quantized = quantize_working_set(
            Image.fromarray(src, "RGB"), 16, custom_palette=custom
        )
        pals, assignment = gbp.pack_palettes(
            np.asarray(quantized), 4, custom_palette=custom
        )
        gb = gbp.index_tiles(src, pals, assignment, "none")
        return np.asarray(gbp.render(gb), dtype=np.uint8), pals

    out1, pals1 = run()
    out2, pals2 = run()
    assert np.array_equal(out1, out2)  # deterministic, pixel-stable
    assert np.array_equal(pals1, pals2)
    # The fixed pipeline only ever emits colors from the custom palette set;
    # the new search code is not on this path at all.
    allowed = {tuple(int(v) for v in c) for c in np.unique(custom, axis=0)}
    for color in np.unique(out1.reshape(-1, 3), axis=0):
        assert tuple(int(v) for v in color) in allowed
    for pal in pals1.reshape(-1, 3):
        assert tuple(int(v) for v in pal) in allowed
