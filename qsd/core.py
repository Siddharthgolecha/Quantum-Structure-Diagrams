from typing import Any, Callable, Dict, List, Tuple, Union

PhaseGauge = Union[bool, None, str, Tuple[str, Any]]

import numpy as np

from .dimensions import resolve_dims

# -----------------------------
# Utilities
# -----------------------------


def ragged_to_padded(rows: List[List[complex]], pad_value=np.nan) -> np.ndarray:
    """Convert ragged rows into a padded 2D complex array.

    Args:
        rows: Ragged list of complex rows.
        pad_value: Value used to pad short rows (nan+0j by default).

    Returns:
        2D complex array with uniform row length.

    Example:
        >>> ragged_to_padded([[1+0j, 2+0j], [3+0j]]).shape
        (2, 2)
    """
    max_len = max((len(r) for r in rows), default=0)
    mat = np.full((len(rows), max_len), pad_value, dtype=complex)
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r
    return mat


def _row_label_sort_key(label: Any) -> Tuple[int, Any]:
    """Return a sort key that stabilizes heterogeneous row labels.

    Example:
        >>> sorted([\"b\", 2, \"a\", 1], key=_row_label_sort_key)
        [1, 2, 'a', 'b']
    """
    if isinstance(label, (int, np.integer)):
        return (0, int(label))
    if isinstance(label, (float, np.floating)):
        return (1, float(label))
    return (2, str(label))


# -----------------------------
# Grouping G and ordering π_r
# -----------------------------

Grouping = Union[str, Callable[[Tuple[int, ...]], Any]]
Ordering = Union[str, Callable[[Tuple[int, ...], int], Any]]
OrderingPerRow = Union[Ordering, Dict[Any, Ordering]]


def _make_grouping_fn(
    dims: List[int], grouping: Grouping
) -> Tuple[str, Callable[[Tuple[int, ...]], Any]]:
    """Create a grouping function G and its resolved name.

    Supported grouping strings:
        - "auto": defaults to "excitation" (nonzero count)
        - "excitation" / "hamming": sum_k 1[level_k != 0]
        - "parity": sum(levels) mod 2
        - "flat" / "none": single row label 0
        - "levelsum": sum(levels) for qudits
        - callable: user-provided G(levels) -> label

    Returns:
        Tuple of (resolved_name, grouping_fn).

    Example:
        >>> name, G = _make_grouping_fn([2, 2], \"parity\")
        >>> name
        'parity'
        >>> G((1, 0))
        1
    """
    if callable(grouping):
        return ("callable", grouping)

    g = grouping
    if g == "auto":
        g = "excitation"

    if g in ("excitation", "hamming"):
        return ("excitation", lambda levels: int(sum(1 for x in levels if x != 0)))

    if g == "parity":
        return ("parity", lambda levels: int(sum(levels) % 2))

    if g in ("flat", "none"):
        return ("flat", lambda levels: 0)

    # Optional explicit legacy grouping (sum of levels) for qudits
    if g in ("levelsum", "sumlevels"):
        return ("levelsum", lambda levels: int(sum(levels)))

    raise ValueError(
        f"Unknown grouping='{grouping}'. "
        "Use auto|excitation|hamming|parity|flat|none|levelsum|callable."
    )


def _make_order_key(
    dims: List[int], ordering: Ordering
) -> Tuple[str, Callable[[Tuple[int, ...], int], Any], bool]:
    """Create an ordering key function for within-row sorting.

    Supported ordering strings:
        - "lex": lexicographic by levels tuple
        - "flat": by flat index
        - "reverse": reverse lex
        - "gray": Gray-code key (qubits only; falls back to lex)
        - callable: user-provided key(levels, flat) -> sortable key

    Returns:
        Tuple of (resolved_name, key_fn, reverse_flag).

    Example:
        >>> name, key_fn, rev = _make_order_key([2, 2], \"lex\")
        >>> name, rev
        ('lex', False)
        >>> key_fn((1, 0), 2)
        (1, 0)
    """
    if callable(ordering):
        return ("callable", ordering, False)

    if ordering == "lex":
        return ("lex", lambda levels, flat: levels, False)

    if ordering == "flat":
        return ("flat", lambda levels, flat: flat, False)

    if ordering == "reverse":
        return ("lex", lambda levels, flat: levels, True)

    if ordering == "gray":
        is_qubits = all(d == 2 for d in dims)
        if not is_qubits:
            # Gray code is not well-defined here; keep behavior deterministic.
            return ("lex", lambda levels, flat: levels, False)

        # For qubits: interpret levels as bits → integer → Gray rank
        def gray_key(levels: Tuple[int, ...], flat: int) -> int:
            g = flat ^ (flat >> 1)
            return int(g)

        return ("gray", gray_key, False)

    raise ValueError("ordering must be lex|flat|reverse|gray|callable.")


# -----------------------------
# Phase gauge + rendering conventions
# -----------------------------

def threshold_for_rendering(psi: np.ndarray, render_eps: float) -> np.ndarray:
    """Zero-out small amplitudes for visualization-only rendering.

    Args:
        psi: Statevector as a flat complex array.
        render_eps: Threshold below which amplitudes are zeroed.

    Returns:
        Thresholded statevector.

    Example:
        >>> import numpy as np
        >>> threshold_for_rendering(np.array([1e-6+0j, 1+0j]), 1e-3)
        array([0.+0.j, 1.+0.j])
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    out = psi.copy()
    out[np.abs(out) < render_eps] = 0.0 + 0.0j
    return out

def apply_phase_gauge(
    psi: np.ndarray,
    gauge: PhaseGauge = "max",
    *,
    eps: float = 1e-15,
) -> np.ndarray:
    """Apply a global phase rotation psi -> psi * exp(-i theta).

    Args:
        psi: Statevector as a flat complex array.
        gauge: Phase gauge strategy ("max", ("index", k), True/False).
        eps: Amplitude threshold for "max" or index gauge.

    Returns:
        Phase-rotated statevector.

    Example:
        >>> import numpy as np
        >>> apply_phase_gauge(np.array([1j, 0]), gauge=\"max\")
        array([1.+0.j, 0.+0.j])
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    if psi.size == 0 or gauge is None or gauge is False:
        return psi

    if gauge is True:
        gauge = "max"

    theta = 0.0

    if gauge == "max":
        mags = np.abs(psi)
        i_star = int(np.argmax(mags))
        if mags[i_star] > eps:
            theta = np.angle(psi[i_star])

    elif isinstance(gauge, tuple) and len(gauge) == 2 and gauge[0] == "index":
        k = int(gauge[1])
        if 0 <= k < psi.size and np.abs(psi[k]) > eps:
            theta = np.angle(psi[k])

    else:
        raise ValueError(f"Unsupported phase gauge: {gauge!r}")

    return psi * np.exp(-1j * theta)


# -----------------------------
# Main construction: build QSD rows
# -----------------------------


def build_qsd_lattice(
    psi: Union[np.ndarray, List[complex]],
    grouping: Grouping = "auto",
    ordering: OrderingPerRow = "lex",
    *,
    normalize_state: bool = False,
    phase_gauge: PhaseGauge = False,
    return_indices: bool = False,
) -> Tuple:
    """Construct QSD row vectors a^(r) from a statevector under (G, {pi_r}).

    Args:
        psi: Statevector coefficients c_i in the computational basis.
        grouping: Grouping function G. See _make_grouping_fn() for supported strings.
        ordering: Within-row ordering pi_r (string, callable, or per-row dict).
        normalize_state: If True, normalize psi to unit norm.
        phase_gauge: Optional global phase convention applied before grouping.
        return_indices: If True, return index rows and permutations.

    Returns:
        ordered: dict[row_label -> list[(flat_index, levels_tuple)]]
        rows: ragged rows of amplitudes a^(r) in the order induced by pi_r
        index_rows, flat_indices, psi_perm: optional extras when return_indices=True

    Example:
        >>> psi = [1, 0, 0, 0]
        >>> ordered, rows, *_ = build_qsd_lattice(psi, grouping=\"hamming\", ordering=\"lex\")
        >>> list(ordered.keys())
        [0]
    """
    dims = resolve_dims(psi)
    psi = np.asarray(psi, dtype=complex).reshape(-1)

    N = int(np.prod(dims))
    if psi.size != N:
        raise ValueError(f"psi has length {psi.size}, but prod(dims)={N}.")

    if normalize_state:
        norm2 = float(np.vdot(psi, psi).real)
        if norm2 <= 0:
            raise ValueError("State has zero norm.")
        psi = psi / np.sqrt(norm2)

    if phase_gauge:
        psi = apply_phase_gauge(psi)

    # levels[i] = (b_{i1},...,b_{in})
    levels_arr = np.stack(np.unravel_index(np.arange(N), dims), axis=1)

    grouping_name, G = _make_grouping_fn(dims, grouping)

    # compute row labels
    if callable(grouping):
        row_labels = np.array([G(tuple(levels_arr[i])) for i in range(N)], dtype=object)
    else:
        # vectorize common groupings
        if grouping_name == "excitation":
            row_labels = np.count_nonzero(levels_arr, axis=1)
        elif grouping_name == "parity":
            row_labels = np.sum(levels_arr, axis=1) % 2
        elif grouping_name == "flat":
            row_labels = np.zeros(N, dtype=int)
        elif grouping_name == "levelsum":
            row_labels = np.sum(levels_arr, axis=1)
        else:
            row_labels = np.array(
                [G(tuple(levels_arr[i])) for i in range(N)], dtype=object
            )

    # group indices by row label
    grouped: Dict[Any, List[int]] = {}
    for i in range(N):
        r = row_labels[i]
        grouped.setdefault(r, []).append(i)

    # determine row order
    row_keys = sorted(grouped.keys(), key=_row_label_sort_key)

    # helper to get ordering per row
    def get_ordering_for_row(rlabel: Any) -> Ordering:
        if isinstance(ordering, dict):
            return ordering.get(rlabel, "lex")
        return ordering

    # order within each group
    ordered: Dict[Any, List[Tuple[int, Tuple[int, ...]]]] = {}
    rows: List[List[complex]] = []
    index_rows: List[List[int]] = []

    for r in row_keys:
        idxs = grouped[r]
        ord_r = get_ordering_for_row(r)
        _, key_fn, reverse_flag = _make_order_key(dims, ord_r)

        # sort by π_r key
        idxs_sorted = sorted(
            idxs,
            key=lambda i: key_fn(tuple(levels_arr[i]), i),
            reverse=reverse_flag,
        )

        ordered[r] = [(i, tuple(levels_arr[i])) for i in idxs_sorted]
        index_rows.append(idxs_sorted)
        rows.append([psi[i] for i in idxs_sorted])

    if not return_indices:
        return ordered, rows

    flat_indices = [i for row in index_rows for i in row]
    psi_perm = psi[flat_indices]
    return ordered, rows, index_rows, flat_indices, psi_perm
