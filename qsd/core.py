from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# -----------------------------
# Utilities
# -----------------------------


def ragged_to_padded(rows: List[List[complex]], pad_value=np.nan) -> np.ndarray:
    """
    Convert ragged complex rows to a padded 2D complex array.
    pad_value defaults to NaN (stored as nan+0j) which many plotting libs treat as empty.
    """
    max_len = max((len(r) for r in rows), default=0)
    mat = np.full((len(rows), max_len), pad_value, dtype=complex)
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r
    return mat


def _row_label_sort_key(label: Any) -> Tuple[int, Any]:
    """Stable sort for heterogeneous row labels."""
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
    """
    Returns (grouping_resolved_name, G(levels)->row_label).

    Supported grouping strings (aligned with your LaTeX):
      - "auto":  defaults to "excitation" (nonzero count)
      - "excitation": sum_k 1[level_k != 0]   (matches Eq. for G_exc)
      - "hamming": alias for "excitation"
      - "parity": (sum_k level_k) mod 2
      - "flat" / "none": single row label 0
      - callable: user-provided G(levels)->label

    NOTE:
      If you ALSO want sum(levels) grouping (old behavior in some code),
      use "levelsum" explicitly (included below) to avoid naming collision.
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
    """
    Returns (ordering_resolved_name, key(levels, flat)->key, reverse_flag).

    Supported ordering strings:
      - "lex": lexicographic by levels tuple
      - "flat": by flat index
      - "reverse": reverse lex
      - "gray": Gray-code key (qubits only; falls back to lex for non-qubits)
      - callable: user-provided key(levels, flat)->sortable key
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


def apply_phase_gauge(psi: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Fix global phase so that the largest-magnitude amplitude is real and >= 0:
        psi -> psi * exp(-i arg(c_{i*})), i* = argmax |c_i|
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    if psi.size == 0:
        return psi
    mags = np.abs(psi)
    i_star = int(np.argmax(mags))
    if mags[i_star] <= eps:
        return psi
    phase = np.angle(psi[i_star])
    return psi * np.exp(-1j * phase)


def threshold_for_rendering(psi: np.ndarray, render_eps: float) -> np.ndarray:
    """
    For visualization only: set small amplitudes to 0 (and thus ignore phase).
    Metrics should be computed on the un-thresholded psi unless explicitly stated otherwise.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    out = psi.copy()
    out[np.abs(out) < render_eps] = 0.0 + 0.0j
    return out


# -----------------------------
# Main construction: build QSD rows
# -----------------------------


def build_qsd_lattice(
    psi: Union[np.ndarray, List[complex]],
    dims: List[int],
    grouping: Grouping = "auto",
    ordering: OrderingPerRow = "lex",
    *,
    normalize_state: bool = False,
    phase_gauge: bool = False,
    return_indices: bool = False,
) -> Tuple:
    """
    Construct QSD row vectors a^(r) from a fixed-basis statevector under (G, {π_r}).

    Parameters
    ----------
    psi : array-like complex, shape (N,)
        Statevector coefficients c_i in the computational basis.
    dims : list[int]
        Local dimensions [d1, ..., dn], so N = prod(dims).
    grouping : str or callable
        Grouping function G. See _make_grouping_fn() for supported strings.
    ordering : str/callable OR dict[row_label -> str/callable]
        Within-row ordering π_r.
        - If a single str/callable is given, it is used for all rows.
        - If a dict is given, it can specify different π_r per row label.
    normalize_state : bool
        If True, normalize psi to unit norm. (Your LaTeX assumes normalized input.)
    phase_gauge : bool
        If True, apply global phase gauge convention for rendering comparability.
    return_indices : bool
        If True, also return per-row flat indices and the row-major permutation.

    Returns
    -------
    ordered : dict[row_label -> list[(flat_index, levels_tuple)]]
        The grouped + ordered basis indices with explicit levels tuples.
    rows : list[list[complex]]
        Ragged rows of amplitudes a^(r) in the order induced by π_r.
    index_rows, flat_indices, psi_perm : optional
        Only if return_indices=True:
          - index_rows: list[list[int]] parallel to rows
          - flat_indices: row-major flattening of index_rows
          - psi_perm: psi permuted into row-major QSD order
    """
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
