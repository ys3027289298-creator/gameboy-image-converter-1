"""Tests for tile_banks.py -- co-occurrence-driven ROM-bank rearrangement.

The module under test deliberately avoids importing gb_pipeline (its image
dependencies are not installed in every environment), so the rebuild
pixel-identity tests use a local GBImage-shaped stand-in carrying the same
field names instead of gb_pipeline.GBImage.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from tile_banks import (
    BankOverflowError,
    _connected_components,
    apply_bank_layout,
    baseline_assignment,
    build_cooccurrence,
    count_switches,
    optimize_bank_layout,
    reorder_patterns,
)


@dataclass
class FakeGBImage:
    """Field-for-field stand-in for gb_pipeline.GBImage."""

    patterns: np.ndarray
    tilemap: np.ndarray
    attrs_palette: np.ndarray
    attrs_hflip: np.ndarray
    attrs_vflip: np.ndarray
    palettes: np.ndarray


def _distinct_patterns(n, seed=0):
    """n distinct 8x8 index patterns (distinct 2bpp bytes), deterministically."""
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        tile = rng.randint(0, 4, size=(8, 8)).astype(np.uint8)
        tile[0, 0] = i % 4  # break any accidental equality across draws
        out.append(tile)
    return np.stack(out)


def _dummy_palettes(n=1):
    return np.tile(
        np.array([[240, 240, 240], [170, 170, 170], [90, 90, 90], [10, 10, 10]], dtype=np.uint8),
        (n, 1, 1),
    )


def _effective_cell(gb, tr, tc):
    """The 8x8 pattern a cell actually draws, flip bits applied."""
    pattern = gb.patterns[int(gb.tilemap[tr, tc])]
    if gb.attrs_hflip[tr, tc]:
        pattern = np.fliplr(pattern)
    if gb.attrs_vflip[tr, tc]:
        pattern = np.flipud(pattern)
    return pattern


def _render(gb):
    """Minimal GBImage renderer (mirrors gb_pipeline.render's per-cell rule)."""
    th, tw = gb.tilemap.shape
    out = np.zeros((th * 8, tw * 8, 3), dtype=np.uint8)
    for tr in range(th):
        for tc in range(tw):
            pattern = _effective_cell(gb, tr, tc)
            pal = gb.palettes[int(gb.attrs_palette[tr, tc])]
            out[tr * 8: tr * 8 + 8, tc * 8: tc * 8 + 8] = pal[pattern]
    return out


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_cooccurrence_undirected_and_self_pairs_omitted():
    seq = np.array([0, 1, 1, 0, 2])
    adj = build_cooccurrence(seq, 3)
    # Pairs: (0,1), (1,1) skipped, (1,0) merges into the same edge, (0,2).
    assert adj[0] == {1: 2, 2: 1}
    assert adj[1] == {0: 2}
    assert adj[2] == {0: 1}


def test_cooccurrence_2d_flattens_in_raster_order_including_row_wrap():
    tilemap = np.array([[0, 1], [2, 3]])  # seq 0,1,2,3 -> wrap pair (1,2) exists
    adj = build_cooccurrence(tilemap.reshape(-1), 4)
    assert set(adj[1]) == {0, 2}
    assert set(adj[2]) == {1, 3}


def test_count_switches_matches_brute_force():
    seq = np.array([0, 1, 2, 2, 1, 0])
    assignment = np.array([0, 0, 1, 1])  # patterns 0,1 -> bank 0; 2 -> bank 1
    # banks: 0,0,1,1,0,0 -> changes at idx 2 and 4
    assert count_switches(seq, assignment) == 2
    assert count_switches(np.array([7]), assignment) == 0
    assert count_switches(np.array([], dtype=np.int64), assignment) == 0


def test_baseline_fills_banks_in_first_appearance_order():
    seq = np.array([3, 3, 0, 1, 2, 0])
    assignment = baseline_assignment(seq, n_patterns=4, bank_capacity=2)
    # First appearance order: 3, 0 fill bank 0; 1, 2 fill bank 1.
    assert assignment.tolist() == [0, 1, 1, 0]
    # Banks drawn: 0,0,0,1,1,0 -> two changes.
    assert count_switches(seq, assignment) == 2


# --------------------------------------------------------------------------
# Optimizer: feasibility, determinism, improvement
# --------------------------------------------------------------------------


def test_optimized_never_worse_than_baseline_and_respects_capacity():
    rng = np.random.RandomState(11)
    for seed in range(20):
        rng_local = np.random.RandomState(seed)
        n_patterns = int(rng_local.randint(4, 18))
        tilemap = rng_local.randint(0, n_patterns, size=(6, 8))
        capacity = int(rng_local.randint(2, 6))
        n_banks = -(-n_patterns // capacity)
        layout = optimize_bank_layout(
            tilemap,
            bank_capacity=capacity,
            n_banks=n_banks,
            n_patterns=n_patterns,
        )
        assert layout.switches <= layout.baseline_switches
        assert int(layout.bank_sizes.max()) <= capacity
        assert layout.assignment.shape == (n_patterns,)
        assert layout.n_patterns == n_patterns
        # Every pattern lands in exactly one of the n_banks banks.
        assert set(layout.assignment.tolist()) <= set(range(n_banks))
        assert np.bincount(layout.assignment, minlength=n_banks).max() <= capacity
        # Cut == switch identity (the adjacency definition guarantee).
        assert layout.cut_edges == layout.switches
        # Allocation table agrees with the assignment.
        for b, members in enumerate(layout.bank_members):
            assert all(layout.assignment[p] == b for p in members)
        assert sum(map(len, layout.bank_members)) == n_patterns


def test_optimized_is_deterministic_across_runs():
    rng = np.random.RandomState(5)
    tilemap = rng.randint(0, 12, size=(8, 10))
    a = optimize_bank_layout(tilemap, bank_capacity=4, n_banks=3)
    b = optimize_bank_layout(tilemap.copy(), bank_capacity=4, n_banks=3)
    assert np.array_equal(a.assignment, b.assignment)
    assert a.switches == b.switches
    assert a.bank_members == b.bank_members


def test_optimized_strictly_improves_on_baseline_for_hub_graph():
    # One hub pattern (4) repeatedly interleaved with four leaf patterns. With
    # capacity 2 the first-appearance baseline keeps the hub far from most
    # leaves, so nearly every cell crosses banks; the co-occurrence-driven
    # layout pulls the hub's strongest neighbors alongside it and strictly cuts
    # the switch count. No cut can be negative, so the result is bounded too.
    tilemap = np.array([
        [0, 1, 2, 3, 4, 0, 4, 1, 4, 2, 4, 3],
        [0, 1, 2, 3, 4, 0, 4, 1, 4, 2, 4, 3],
    ])
    layout = optimize_bank_layout(tilemap, bank_capacity=2, n_banks=3)
    assert layout.baseline_switches == 19
    assert layout.switches < layout.baseline_switches
    assert layout.switches >= 0
    assert int(layout.bank_sizes.max()) <= 2


def test_optimized_groups_strongly_co_occurring_patterns_into_one_bank():
    # Two strongly linked pairs {0,1} and {2,3}, with pattern 0 also bridging
    # to 2/3. The first-appearance baseline (banks 0,0,1,1) cuts the strong
    # {0,1} tie only weakly here; the optimizer must at minimum match it, keep
    # every bank within capacity, and bank 0's strongest neighbor alongside it.
    tilemap = np.array([
        [0, 1, 0, 2, 0, 3, 0, 1],
        [1, 0, 2, 0, 3, 0, 1, 0],
        [0, 1, 0, 2, 0, 3, 0, 1],
    ])
    adj = build_cooccurrence(tilemap.reshape(-1), 4)
    assert max(adj[0], key=adj[0].get) == 1  # 1 is pattern 0's strongest tie
    layout = optimize_bank_layout(tilemap, bank_capacity=2, n_banks=2)
    assert layout.switches <= layout.baseline_switches
    assert int(layout.bank_sizes.max()) <= 2
    assert sum(map(len, layout.bank_members)) == 4
    pair_banks = layout.assignment[[0, 1]]
    assert pair_banks[0] == pair_banks[1]  # strongest pair shares a bank


def test_single_bank_layout_is_switch_free():
    tilemap = np.arange(6).reshape(2, 3)
    layout = optimize_bank_layout(tilemap, bank_capacity=10)
    assert layout.n_banks == 1
    assert layout.switches == 0
    assert layout.baseline_switches == 0


def test_n_banks_none_uses_minimum_required():
    tilemap = np.array([[0, 1, 2], [3, 4, 5]])
    layout = optimize_bank_layout(tilemap, bank_capacity=2)
    assert layout.n_banks == 3
    assert layout.n_banks_requested is None
    assert layout.banks_used == 3


def test_empty_tilemap_is_handled():
    layout = optimize_bank_layout(
        np.zeros((0, 0), dtype=np.int64), bank_capacity=4, n_banks=2
    )
    assert layout.n_patterns == 0
    assert layout.switches == 0
    assert layout.bank_members == []
    assert layout.warnings == []


# --------------------------------------------------------------------------
# Overflow: explicit errors and degrade-with-warning policy
# --------------------------------------------------------------------------


def test_overflow_strict_raises():
    tilemap = np.array([[0, 1, 2], [3, 4, 5]])
    with pytest.raises(BankOverflowError, match="needs 3 banks"):
        optimize_bank_layout(tilemap, bank_capacity=2, n_banks=2)


def test_overflow_expand_degrades_with_warning():
    tilemap = np.array([[0, 1, 2], [3, 4, 5]])
    layout = optimize_bank_layout(
        tilemap, bank_capacity=2, n_banks=2, on_overflow="expand"
    )
    assert layout.n_banks == 3
    assert layout.n_banks_requested == 2
    assert len(layout.warnings) == 1
    assert "expanded to 3 banks" in layout.warnings[0]
    assert layout.switches <= layout.baseline_switches


def test_invalid_arguments_rejected():
    tilemap = np.array([[0, 1]])
    with pytest.raises(ValueError):
        optimize_bank_layout(tilemap, bank_capacity=0)
    with pytest.raises(ValueError):
        optimize_bank_layout(tilemap, bank_capacity=4, n_banks=0)
    with pytest.raises(ValueError):
        optimize_bank_layout(tilemap, bank_capacity=4, on_overflow="nope")


# --------------------------------------------------------------------------
# allow_cross_bank_refs=False: component packing, switch-free guarantee
# --------------------------------------------------------------------------


def test_no_cross_refs_packs_components_and_is_switch_free():
    # The tilemap only references patterns {0, 1, 2}; pattern 3 never appears,
    # so the co-occurrence graph has two components: {0,1,2} and isolated {3}.
    # The referenced component must land whole in one bank; the unused tile
    # fills the leftover slot. Capacity 3, two banks available.
    tilemap = np.array([
        [0, 1, 0, 2, 1, 2, 0],
        [1, 0, 2, 0, 2, 1, 2],
    ])
    adj = build_cooccurrence(tilemap.reshape(-1), 4)
    assert [tuple(c) for c in _connected_components(adj, 4)] == [(0, 1, 2), (3,)]
    layout = optimize_bank_layout(
        tilemap, bank_capacity=3, n_banks=2, allow_cross_bank_refs=False
    )
    assert layout.switches == 0
    assert layout.allow_cross_bank_refs is False
    # No co-occurrence edge is cut: referenced patterns share one bank.
    ref_bank = layout.assignment[0]
    assert layout.assignment[1] == ref_bank
    assert layout.assignment[2] == ref_bank
    assert layout.banks_used <= 2


def test_no_cross_refs_strict_raises_when_component_exceeds_capacity():
    # All four patterns are referenced -> one connected component of 4, but a
    # bank holds only 3: the no-cross-reference restriction is impossible.
    tilemap = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    with pytest.raises(BankOverflowError, match="allow_cross_bank_refs"):
        optimize_bank_layout(
            tilemap, bank_capacity=3, n_banks=2, allow_cross_bank_refs=False
        )


def test_no_cross_refs_expand_relaxes_restriction_with_warning():
    tilemap = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    layout = optimize_bank_layout(
        tilemap,
        bank_capacity=3,
        n_banks=2,
        allow_cross_bank_refs=False,
        on_overflow="expand",
    )
    assert layout.allow_cross_bank_refs is True
    assert any("relaxed" in w for w in layout.warnings)
    assert layout.switches <= layout.baseline_switches


def test_no_cross_refs_raises_when_components_do_not_fit_available_banks():
    # Five pattern slots exist; the sequence only ever references {0,1,2}, so
    # the co-occurrence graph has the {0,1,2} component plus isolated nodes
    # {3} and {4}. With capacity 3 and a single bank the five nodes cannot be
    # packed while keeping the component intact.
    seq = np.array([0, 1, 0, 2, 1, 2, 0, 1])
    adj = build_cooccurrence(seq, 5)
    components = [tuple(c) for c in _connected_components(adj, 5)]
    assert components == [(0, 1, 2), (3,), (4,)]
    with pytest.raises(BankOverflowError):
        optimize_bank_layout(
            seq,
            bank_capacity=3,
            n_banks=1,
            n_patterns=5,
            allow_cross_bank_refs=False,
        )


# --------------------------------------------------------------------------
# Rebuild: storage relayout changes no rendered pixel
# --------------------------------------------------------------------------


def _make_fake_gb(tilemap, n_patterns, seed=0):
    th, tw = tilemap.shape
    patterns = _distinct_patterns(n_patterns, seed)
    rng = np.random.RandomState(seed + 100)
    return FakeGBImage(
        patterns=patterns,
        tilemap=tilemap.astype(np.int32),
        attrs_palette=rng.randint(0, 2, size=(th, tw)).astype(np.uint8),
        attrs_hflip=rng.randint(0, 2, size=(th, tw)).astype(bool),
        attrs_vflip=rng.randint(0, 2, size=(th, tw)).astype(bool),
        palettes=_dummy_palettes(2),
    )


def test_reorder_patterns_preserves_effective_cells():
    tilemap = np.array([
        [0, 4, 1, 5, 2, 6],
        [4, 0, 5, 1, 6, 2],
        [0, 4, 1, 5, 2, 6],
    ])
    gb = _make_fake_gb(tilemap, n_patterns=7)
    layout = optimize_bank_layout(
        tilemap, bank_capacity=3, n_banks=3, n_patterns=7
    )
    new_gb = apply_bank_layout(gb, layout)

    # Storage is actually rearranged and grouped contiguously by bank.
    assert not np.array_equal(new_gb.tilemap, gb.tilemap)
    stored_banks = layout.assignment[np.argsort(layout.assignment, kind="stable")]
    assert np.array_equal(np.sort(stored_banks), stored_banks)

    # Every cell still draws the exact same effective pattern bytes.
    th, tw = tilemap.shape
    for tr in range(th):
        for tc in range(tw):
            before = _effective_cell(gb, tr, tc)
            after = _effective_cell(new_gb, tr, tc)
            assert np.array_equal(before, after)
    # And therefore the full render is pixel-identical.
    assert np.array_equal(_render(gb), _render(new_gb))
    # Non-pattern attributes are untouched.
    assert np.array_equal(new_gb.attrs_palette, gb.attrs_palette)
    assert np.array_equal(new_gb.attrs_hflip, gb.attrs_hflip)
    assert np.array_equal(new_gb.attrs_vflip, gb.attrs_vflip)
    assert np.array_equal(new_gb.palettes, gb.palettes)


def test_reorder_is_a_bijection_and_idempotent_under_same_layout():
    tilemap = np.random.RandomState(3).randint(0, 10, size=(4, 5))
    patterns = _distinct_patterns(10, seed=3)
    layout = optimize_bank_layout(tilemap, bank_capacity=4)
    new_patterns, new_tilemap, old_index = reorder_patterns(
        patterns, tilemap, layout.assignment
    )
    # old_index is a permutation and pattern multiset is preserved.
    assert sorted(old_index.tolist()) == list(range(10))
    assert {p.tobytes() for p in new_patterns} == {p.tobytes() for p in patterns}
    # Every new tilemap reference resolves to the pattern it did before.
    assert np.array_equal(
        new_patterns[new_tilemap],
        patterns[tilemap],
    )
    # Re-applying the same layout is a storage-equivalent relayout: the tilemap
    # still resolves to exactly the same pattern at every cell.
    again_patterns, again_tilemap, _ = reorder_patterns(
        new_patterns, new_tilemap, layout.assignment
    )
    assert np.array_equal(again_patterns[again_tilemap], new_patterns[new_tilemap])
    assert {p.tobytes() for p in again_patterns} == {p.tobytes() for p in new_patterns}
    # And the natural idempotence: running the optimizer again on the rewritten
    # tilemap yields the same switch count and bank sizes.
    again_layout = optimize_bank_layout(
        again_tilemap, bank_capacity=layout.bank_capacity
    )
    assert again_layout.switches == layout.switches
    assert np.array_equal(again_layout.bank_sizes, layout.bank_sizes)


def test_reorder_rejects_index_out_of_range():
    patterns = _distinct_patterns(3)
    tilemap = np.array([[0, 3]])
    assignment = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        reorder_patterns(patterns, tilemap, assignment)
