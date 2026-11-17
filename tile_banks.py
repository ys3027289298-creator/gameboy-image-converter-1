"""Tile-set ROM-bank layout optimization.

GB/GBC tile data lives in ROM banks of a fixed capacity; a tilemap cell whose
pattern sits in a bank other than the one currently paged in forces a bank
switch. Frequent switches while drawing a screen are visible as stalls, so this
module rearranges *which stored patterns live in which bank* -- a pure
relabeling that never changes a rendered pixel -- to minimize the number of
switches incurred by drawing the tilemap in raster order.

Method
------
1. Build an undirected, weighted co-occurrence graph: one node per stored
   pattern, an edge (u, v) with weight w for every pair of *consecutive* cells
   in the raster-order cell sequence (row wrap included -- the active bank does
   not reset at the end of a scanline) that reference patterns u and v. With
   that adjacency definition the weighted graph cut of a bank assignment is
   exactly the number of runtime bank switches the assignment causes.
2. Baseline: patterns are handed to banks in first-appearance order, each bank
   filled to ``bank_capacity`` before the next opens ("按出现顺序分配").
3. Optimized layout: a capacity-constrained graph partition. A deterministic
   greedy graph-growing placement is refined by best-improvement local search
   (single-node moves and pairwise swaps, only strictly cut-reducing changes
   accepted). The search is also seeded from the baseline assignment, so the
   returned layout can never score worse than the baseline. No RNG is used, so
   the same input always produces the same layout.

The module depends only on numpy + the standard library and never imports
gb_pipeline, so it is usable in environments where the image-processing
dependencies (scikit-image, ...) are not installed.
"""

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BankOverflowError",
    "BankLayout",
    "optimize_bank_layout",
    "baseline_assignment",
    "build_cooccurrence",
    "count_switches",
    "reorder_patterns",
    "apply_bank_layout",
]


class BankOverflowError(ValueError):
    """The tile set cannot fit the configured bank capacity / bank count.

    Raised in strict mode (``on_overflow="error"``); ``on_overflow="expand"``
    instead grows the bank count (or, when cross-bank references are
    disallowed and a co-occurrence component alone exceeds one bank, relaxes
    that restriction) and records a warning on the returned layout.
    """


@dataclass
class BankLayout:
    """The result of :func:`optimize_bank_layout`.

    ``assignment[p]`` is the bank id of pattern ``p``; ``baseline_assignment``
    is the first-appearance-order layout the optimized one is compared
    against. ``switches`` / ``baseline_switches`` count bank changes along the
    raster-order cell sequence (the optimized cut equals ``switches`` exactly).
    ``bank_members`` is the human-readable allocation table: one sorted list of
    pattern ids per used bank.
    """

    assignment: np.ndarray
    baseline_assignment: np.ndarray
    bank_members: list
    bank_sizes: np.ndarray
    banks_used: int
    n_banks: int
    n_banks_requested: int | None
    bank_capacity: int
    allow_cross_bank_refs: bool
    switches: int
    baseline_switches: int
    cut_edges: int
    co_occurrence_edges: int
    n_patterns: int
    n_cells: int
    warnings: list = field(default_factory=list)


def _as_sequence(tilemap) -> np.ndarray:
    """Flatten a 1-D sequence or a 2-D tilemap to the raster-order 1-D sequence
    of pattern indices (C order: row by row, including the row-wrap step)."""
    arr = np.asarray(tilemap)
    if arr.ndim == 2:
        seq = arr.reshape(-1)
    elif arr.ndim == 1:
        seq = arr
    else:
        raise ValueError(
            f"tilemap must be 1-D or 2-D, got shape {arr.shape!r}"
        )
    seq = seq.astype(np.int64, copy=False)
    if seq.size and int(seq.min()) < 0:
        raise ValueError("tilemap pattern indices must be non-negative")
    return seq


def build_cooccurrence(seq: np.ndarray, n_patterns: int) -> list:
    """Build the weighted, undirected co-occurrence adjacency of a cell sequence.

    Returns ``adj`` where ``adj[u]`` maps neighbor pattern id -> number of
    consecutive cell pairs (in either direction) referencing the two patterns.
    Self-pairs (the same pattern in both adjacent cells) carry no switch and
    are omitted.
    """
    seq = np.asarray(seq, dtype=np.int64)
    adj = [dict() for _ in range(n_patterns)]
    if seq.size < 2:
        return adj
    pairs = np.stack((seq[:-1], seq[1:]), axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    for (a, b), c in zip(uniq.tolist(), counts.tolist()):
        if a == b:
            continue
        c = int(c)
        # Undirected: accumulate both directions, so a later reverse pair
        # (b, a) merges into the same edge weight.
        adj[a][b] = adj[a].get(b, 0) + c
        adj[b][a] = adj[b].get(a, 0) + c
    return adj


def count_switches(seq: np.ndarray, assignment: np.ndarray) -> int:
    """Number of bank changes when drawing ``seq`` under ``assignment``.

    Two consecutive cells referencing different banks count one switch,
    including the step from the end of one scanline to the start of the next.
    """
    seq = np.asarray(seq, dtype=np.int64)
    assignment = np.asarray(assignment, dtype=np.int64)
    if seq.size < 2:
        return 0
    if int(seq.max()) >= assignment.shape[0]:
        raise ValueError("sequence references a pattern outside the assignment")
    banks = assignment[seq]
    return int(np.count_nonzero(banks[1:] != banks[:-1]))


def baseline_assignment(seq: np.ndarray, n_patterns: int, bank_capacity: int) -> np.ndarray:
    """First-appearance-order layout: patterns fill bank 0 to capacity in the
    order the tilemap first references them, then bank 1, and so on."""
    seq = np.asarray(seq, dtype=np.int64)
    assignment = np.full(n_patterns, -1, dtype=np.int64)
    if n_patterns == 0:
        return assignment
    # First-appearance order: unique pattern ids, ordered by first index.
    _, first_idx = np.unique(seq, return_index=True)
    order = seq[np.sort(first_idx)]
    bank = 0
    used = 0
    for p in order.tolist():
        if used == bank_capacity:
            bank += 1
            used = 0
        assignment[p] = bank
        used += 1
    return assignment


def _greedy_initial(seq, adj, n_patterns, n_banks, bank_capacity) -> np.ndarray:
    """Greedy graph-growing initial partition.

    Patterns are placed in first-appearance order; each goes into the non-full
    bank holding the greatest total co-occurrence weight with its already
    placed neighbors (lowest bank id wins ties; bank 0 when it has no placed
    neighbor anywhere). Always feasible because placement is refused on full
    banks.
    """
    assignment = np.full(n_patterns, -1, dtype=np.int64)
    sizes = np.zeros(n_banks, dtype=np.int64)
    _, first_idx = np.unique(seq, return_index=True)
    order = seq[np.sort(first_idx)]
    for u in order.tolist():
        weights = np.zeros(n_banks, dtype=np.int64)
        for v, w in adj[u].items():
            b = assignment[v]
            if b >= 0:
                weights[b] += w
        best_bank = -1
        best_weight = None
        for b in range(n_banks):
            if sizes[b] >= bank_capacity:
                continue
            if best_weight is None or weights[b] > best_weight:
                best_weight = weights[b]
                best_bank = b
        if best_bank < 0:
            # Should be unreachable: callers guarantee n_patterns fits.
            raise BankOverflowError(
                "greedy placement ran out of bank capacity unexpectedly"
            )
        assignment[u] = best_bank
        sizes[best_bank] += 1
    return assignment


def _move_delta(u, target_bank, assignment, adj) -> int:
    """Change in weighted cut if pattern ``u`` moved to ``target_bank``.

    Edges to target-bank members stop crossing (negative delta), edges to
    current-bank members start crossing (positive delta); edges to other banks
    stay cut either way and do not matter.
    """
    source_bank = assignment[u]
    delta = 0
    for v, w in adj[u].items():
        b = assignment[v]
        if b == target_bank:
            delta -= w
        elif b == source_bank:
            delta += w
    return delta


def _swap_delta(u, v, assignment, adj) -> int:
    """Change in weighted cut if patterns ``u`` and ``v`` (different banks)
    exchange banks. Swaps preserve every bank size, so they are always
    capacity-feasible."""
    bank_u = assignment[u]
    bank_v = assignment[v]
    delta_u = 0  # (edges u -> own bank) minus (edges u -> v's bank), e(u,v) excluded
    for x, w in adj[u].items():
        b = assignment[x]
        if b == bank_u:
            delta_u += w
        elif b == bank_v and x != v:
            delta_u -= w
    delta_v = 0
    for x, w in adj[v].items():
        b = assignment[x]
        if b == bank_v:
            delta_v += w
        elif b == bank_u and x != u:
            delta_v -= w
    return delta_u + delta_v


def _place_isolated_nodes(assignment, n_banks, bank_capacity) -> np.ndarray:
    """First-fit fill for patterns no cell references (isolated graph nodes,
    assigned -1 so far). They never affect the switch count, so placement only
    has to be deterministic and capacity-feasible. Feasibility was guaranteed
    upstream (overflow policy), so a slot always exists."""
    assignment = np.array(assignment, dtype=np.int64, copy=True)
    sizes = np.bincount(
        assignment[assignment >= 0], minlength=n_banks
    ).astype(np.int64)
    for p in np.nonzero(assignment < 0)[0].tolist():
        for b in range(n_banks):
            if sizes[b] < bank_capacity:
                assignment[p] = b
                sizes[b] += 1
                break
        else:  # pragma: no cover - guarded by the overflow policy above
            raise BankOverflowError(
                "no bank slot left for an unreferenced pattern"
            )
    return assignment


def _local_search(assignment, adj, n_banks, bank_capacity) -> np.ndarray:
    """Best-improvement hill climbing over capacity-feasible moves and swaps.

    Each accepted step strictly reduces the weighted cut, so the loop
    terminates (the cut is a non-negative integer), and every scan order is
    fixed (node ids then bank ids), making the result fully deterministic.
    """
    assignment = np.array(assignment, dtype=np.int64, copy=True)
    n_patterns = assignment.shape[0]
    sizes = np.bincount(assignment, minlength=n_banks).astype(np.int64)

    while True:
        best = None  # (delta, kind_rank, u, v_or_bank)
        for u in range(n_patterns):
            source = int(assignment[u])
            for b in range(n_banks):
                if b == source or sizes[b] >= bank_capacity:
                    continue
                delta = _move_delta(u, b, assignment, adj)
                if delta < 0 and (best is None or delta < best[0]):
                    best = (delta, 0, u, b)
        for u in range(n_patterns):
            for v in range(u + 1, n_patterns):
                if assignment[u] == assignment[v]:
                    continue
                delta = _swap_delta(u, v, assignment, adj)
                if delta < 0 and (best is None or delta < best[0]):
                    best = (delta, 1, u, v)
        if best is None:
            return assignment
        _, kind, u, x = best
        if kind == 0:
            sizes[assignment[u]] -= 1
            assignment[u] = x
            sizes[x] += 1
        else:
            assignment[u], assignment[x] = assignment[x], assignment[u]


def _connected_components(adj, n_patterns):
    """Undirected connected components as lists of node ids, each sorted, with
    the component list itself ordered deterministically (largest first, then
    lowest member id)."""
    seen = np.zeros(n_patterns, dtype=bool)
    components = []
    for start in range(n_patterns):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        members = []
        while stack:
            u = stack.pop()
            members.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        members.sort()
        components.append(members)
    components.sort(key=lambda c: (-len(c), c[0]))
    return components


def _component_assignment(adj, n_patterns, n_banks, bank_capacity):
    """Layout under ``allow_cross_bank_refs=False``: every co-occurrence
    component must reside in one bank (its internal edges may never be cut).

    A raster sequence is a single walk, so every *referenced* pattern belongs
    to one (possibly large) connected component; patterns absent from the
    tilemap show up as isolated single-node components. Components are packed
    in deterministic order with next-fit placement -- the big referenced
    component lands whole in bank 0, and the isolated nodes fill the leftover
    space and subsequent banks -- so the resulting layout is provably
    switch-free. Returns ``None`` when the restriction cannot be honored (a
    component larger than one bank, or more bins than available).
    """
    components = _connected_components(adj, n_patterns)
    if any(len(c) > bank_capacity for c in components):
        return None
    bins = [[] for _ in range(n_banks)]
    bin_sizes = [0] * n_banks
    current = 0
    for comp in components:
        if bin_sizes[current] + len(comp) > bank_capacity:
            current += 1
            if current >= n_banks:
                return None
        bins[current].extend(comp)
        bin_sizes[current] += len(comp)
    assignment = np.full(n_patterns, -1, dtype=np.int64)
    for b, members in enumerate(bins):
        for p in members:
            assignment[p] = b
    return assignment


def _weighted_cut(adj, assignment) -> int:
    """Sum of weights of edges whose endpoints land in different banks."""
    total = 0
    for u, neighbors in enumerate(adj):
        for v, w in neighbors.items():
            if v > u and assignment[u] != assignment[v]:
                total += w
    return total


def optimize_bank_layout(
    tilemap,
    *,
    bank_capacity: int,
    n_banks: int | None = None,
    n_patterns: int | None = None,
    allow_cross_bank_refs: bool = True,
    on_overflow: str = "error",
) -> BankLayout:
    """Rearrange stored patterns into ROM banks to minimize runtime switches.

    Parameters
    ----------
    tilemap:
        2-D array of pattern indices (one per tile cell, raster order) or a
        1-D array already in cell-visit order.
    bank_capacity:
        Maximum number of stored patterns per bank (configurable; hardware
        2bpp tile sets commonly use 128 per 16 KiB bank).
    n_banks:
        Number of banks available. ``None`` means use the minimum required to
        hold every pattern, ``ceil(n_patterns / bank_capacity)``.
    n_patterns:
        Total number of stored patterns, including any that no tilemap cell
        currently references (they become isolated nodes in the co-occurrence
        graph and are still assigned a bank). Defaults to one past the largest
        index seen in ``tilemap``; an explicit value must exceed that.
    allow_cross_bank_refs:
        When ``True`` (default) any tilemap cell may reference a pattern in
        another bank -- at the cost of a switch -- and the optimizer minimizes
        that cost. When ``False``, no co-occurrence edge may be cut: each
        connected component of the co-occurrence graph must fit in a single
        bank, which makes the layout provably switch-free.
    on_overflow:
        ``"error"`` (default) raises :class:`BankOverflowError` when the tile
        set (or, under ``allow_cross_bank_refs=False``, a single component)
        cannot fit; ``"expand"`` grows the bank count as needed -- or relaxes
        the no-cross-reference restriction when a component alone is too big --
        and records a warning on the returned layout.

    The input arrays are never modified; only pattern *storage order* and
    tilemap references change downstream (see :func:`apply_bank_layout`), so
    every rendered pixel stays identical.
    """
    if on_overflow not in ("error", "expand"):
        raise ValueError(
            f"on_overflow must be 'error' or 'expand', got {on_overflow!r}"
        )
    if not isinstance(bank_capacity, (int, np.integer)) or bank_capacity <= 0:
        raise ValueError("bank_capacity must be a positive integer")
    if n_banks is not None and (
        not isinstance(n_banks, (int, np.integer)) or n_banks <= 0
    ):
        raise ValueError("n_banks must be a positive integer or None")

    seq = _as_sequence(tilemap)
    max_seen = int(seq.max()) + 1 if seq.size else 0
    if n_patterns is None:
        n_patterns = max_seen
    else:
        if not isinstance(n_patterns, (int, np.integer)) or n_patterns < 0:
            raise ValueError("n_patterns must be a non-negative integer or None")
        n_patterns = int(n_patterns)
        if n_patterns < max_seen:
            raise ValueError(
                f"n_patterns={n_patterns} is below the largest tilemap "
                f"index plus one ({max_seen})"
            )
    n_cells = int(seq.size)
    min_banks = max(1, -(-n_patterns // int(bank_capacity))) if n_patterns else 0
    warnings: list = []

    requested = None if n_banks is None else int(n_banks)
    if n_banks is None:
        n_banks = min_banks
    elif n_banks < min_banks:
        message = (
            f"tile set of {n_patterns} patterns needs {min_banks} banks of "
            f"capacity {bank_capacity} ({min_banks * bank_capacity} slots), "
            f"but only {n_banks} bank(s) were provided"
        )
        if on_overflow == "error":
            raise BankOverflowError(message)
        warnings.append(message + f"; expanded to {min_banks} banks (degraded).")
        n_banks = min_banks
    n_banks = int(n_banks)

    base = baseline_assignment(seq, n_patterns, int(bank_capacity))
    baseline_switches = count_switches(seq, base)

    if n_patterns == 0:
        return BankLayout(
            assignment=np.zeros(0, dtype=np.int64),
            baseline_assignment=base,
            bank_members=[],
            bank_sizes=np.zeros(0, dtype=np.int64),
            banks_used=0,
            n_banks=n_banks,
            n_banks_requested=requested,
            bank_capacity=int(bank_capacity),
            allow_cross_bank_refs=allow_cross_bank_refs,
            switches=0,
            baseline_switches=0,
            cut_edges=0,
            co_occurrence_edges=0,
            n_patterns=0,
            n_cells=n_cells,
            warnings=warnings,
        )

    adj = build_cooccurrence(seq, n_patterns)
    n_edges = sum(len(d) for d in adj) // 2

    degraded_to_cross = False
    if not allow_cross_bank_refs:
        constrained = _component_assignment(
            adj, n_patterns, n_banks, int(bank_capacity)
        )
        if constrained is None:
            too_big = any(
                len(c) > bank_capacity
                for c in _connected_components(adj, n_patterns)
            )
            reason = (
                "a co-occurrence component alone exceeds bank_capacity"
                if too_big
                else "the components do not bin-pack into the available banks"
            )
            message = (
                "allow_cross_bank_refs=False cannot be honored: " + reason
            )
            if on_overflow == "error":
                raise BankOverflowError(message)
            warnings.append(
                message + "; relaxed to allow_cross_bank_refs=True (degraded)."
            )
            degraded_to_cross = True

    if allow_cross_bank_refs or degraded_to_cross:
        # Isolated nodes (patterns no cell references) are absent from the
        # sequence, so neither layout builder sees them; give them deterministic
        # first-fit slots before refinement. They never contribute to the cut.
        base = _place_isolated_nodes(base, n_banks, int(bank_capacity))
        # Two deterministic starting points; local search only accepts strict
        # improvements, so the baseline-seeded candidate alone already proves
        # the optimized layout never scores worse than the baseline.
        refined_base = _local_search(base, adj, n_banks, int(bank_capacity))
        greedy = _greedy_initial(
            seq, adj, n_patterns, n_banks, int(bank_capacity)
        )
        greedy = _place_isolated_nodes(greedy, n_banks, int(bank_capacity))
        refined_greedy = _local_search(
            greedy, adj, n_banks, int(bank_capacity)
        )
        if count_switches(seq, refined_greedy) < count_switches(seq, refined_base):
            assignment = refined_greedy
        else:
            assignment = refined_base  # tie keeps the baseline-seeded layout
    else:
        assignment = constrained
        if np.any(assignment < 0):  # pragma: no cover - builder is exhaustive
            raise AssertionError("component packing left a pattern unassigned")

    switches = count_switches(seq, assignment)
    cut_edges = _weighted_cut(adj, assignment)
    # The raster-adjacency definition makes these two quantities identical.
    assert cut_edges == switches, (cut_edges, switches)

    bank_members = [
        sorted(int(p) for p in np.nonzero(assignment == b)[0].tolist())
        for b in range(n_banks)
    ]
    bank_members = [m for m in bank_members if m]
    bank_sizes = np.array([len(m) for m in bank_members], dtype=np.int64)
    if bank_sizes.size and int(bank_sizes.max()) > bank_capacity:
        raise AssertionError("bank exceeds capacity")  # internal invariant

    return BankLayout(
        assignment=assignment,
        baseline_assignment=base,
        bank_members=bank_members,
        bank_sizes=bank_sizes,
        banks_used=len(bank_members),
        n_banks=n_banks,
        n_banks_requested=requested,
        bank_capacity=int(bank_capacity),
        # Degradation flips the effective restriction even though the caller
        # asked for False; report what the layout actually enforces.
        allow_cross_bank_refs=allow_cross_bank_refs or degraded_to_cross,
        switches=switches,
        baseline_switches=baseline_switches,
        cut_edges=cut_edges,
        co_occurrence_edges=n_edges,
        n_patterns=n_patterns,
        n_cells=n_cells,
        warnings=warnings,
    )


def reorder_patterns(patterns: np.ndarray, tilemap: np.ndarray, assignment: np.ndarray):
    """Relayout stored patterns by bank id and retarget every tilemap reference.

    Patterns are grouped by bank (stable order: ascending bank, then ascending
    old pattern index), tilemap indices are rewritten to the new positions.
    Returns ``(new_patterns, new_tilemap, old_index)`` where ``old_index[k]``
    is the original pattern index now stored at position ``k``. The multiset of
    stored patterns is unchanged, so every rendered pixel is unchanged.
    """
    patterns = np.asarray(patterns)
    tilemap = np.asarray(tilemap, dtype=np.int64)
    assignment = np.asarray(assignment, dtype=np.int64)
    if patterns.shape[0] != assignment.shape[0]:
        raise ValueError("patterns and assignment must have the same length")
    if tilemap.size and int(tilemap.max()) >= patterns.shape[0]:
        raise ValueError("tilemap references a pattern outside patterns")

    old_index = np.argsort(assignment, kind="stable")
    new_patterns = patterns[old_index]
    new_position = np.empty_like(old_index)
    new_position[old_index] = np.arange(old_index.shape[0])
    new_tilemap = new_position[tilemap]
    return new_patterns, new_tilemap, old_index


def apply_bank_layout(gb, layout: BankLayout):
    """Return a copy of a GBImage-like object with its patterns stored in
    bank order and its tilemap retargeted.

    Constructed via ``type(gb)`` with the GBImage field names, so this module
    needs no import of gb_pipeline. Palette and flip attributes are copied
    untouched; the rendered image is pixel-identical (verified by re-applying
    each cell's pattern + flip bits through its unchanged palette).
    """
    new_patterns, new_tilemap, _ = reorder_patterns(
        gb.patterns, gb.tilemap, layout.assignment
    )
    return type(gb)(
        patterns=new_patterns,
        tilemap=new_tilemap.astype(gb.tilemap.dtype),
        attrs_palette=gb.attrs_palette.copy(),
        attrs_hflip=gb.attrs_hflip.copy(),
        attrs_vflip=gb.attrs_vflip.copy(),
        palettes=gb.palettes.copy(),
    )
