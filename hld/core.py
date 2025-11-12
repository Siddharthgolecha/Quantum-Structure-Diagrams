from itertools import product

import numpy as np


def group_basis_states(dims, grouping="hamming"):
    """
    Returns a dictionary mapping group labels (rows) → list of basis state indices.
    grouping:
        "hamming" → number of non-zero qudit values per basis state
        or a callable f(bitstring)->group
    """
    basis_indices = range(np.prod(dims))
    states = [np.unravel_index(i, dims) for i in basis_indices]

    if grouping == "hamming":
        def G(s): return sum(1 for x in s if x != 0)
    elif callable(grouping):
        G = grouping
    else:
        raise ValueError("grouping must be 'hamming' or a callable")

    groups = {}
    for i, s in zip(basis_indices, states):
        g = G(s)
        groups.setdefault(g, []).append((i, s))
    return dict(sorted(groups.items()))


def order_basis_states(grouped_states, ordering="lex"):
    """
    Given row→[(index, basis-vector), ...], impose per-row ordering.
    """
    ordered = {}
    for r, items in grouped_states.items():
        if ordering == "lex":
            ordered[r] = sorted(items, key=lambda t: t[1])
        elif ordering == "gray":
            # placeholder lex = default
            ordered[r] = sorted(items, key=lambda t: t[1])
        else:
            raise ValueError("ordering must be 'lex' or 'gray'")
    return ordered


def build_hld_lattice(psi, dims, grouping="hamming", ordering="lex", return_indices=False):
    """
    Convert a statevector `psi` into a ragged 2D array of complex amplitudes
    grouped by subspace (e.g., excitation number) and ordered within each group.

    Parameters
    ----------
    psi : np.ndarray[complex]
        Statevector as a 1D complex array of length ∏(dims).
    dims : list[int]
        Dimensions of each subsystem (e.g. [2,2,2] for 3 qubits).
    grouping : str, optional
        How to group computational basis states. Supported:
        - "hamming" : by number of nonzero digits (qubit excitation number)
        - "none" or "flat" : single group containing all states.
    ordering : str, optional
        How to order within each group. Supported:
        - "lex" : lexicographic ascending order (default)
        - "reverse" : descending lexicographic order
    return_indices : bool, optional
        If True, also return a parallel 2D list of the original linear basis indices.

    Returns
    -------
    ordered : dict[int, list[tuple[int, str]]]
        Mapping from group key → list of (basis_index, bitstring).
    rows : list[list[complex]]
        2D ragged list of amplitudes grouped and ordered.
    index_rows : list[list[int]], optional
        Parallel 2D list of original indices (only if `return_indices=True`).
    """
    psi = np.asarray(psi, dtype=complex)
    n_states = len(psi)

    # ---------------- Build computational basis strings ----------------
    def int_to_str(idx):
        digits = []
        n = idx
        for base in reversed(dims):
            digits.append(str(n % base))
            n //= base
        digits.reverse()
        return "".join(digits)

    # ---------------- Group by rule ----------------
    grouped = {}
    for i in range(n_states):
        basis_str = int_to_str(i)
        if grouping == "hamming":
            # excitation number (nonzero digits)
            key = sum(ch != "0" for ch in basis_str)
        else:
            key = 0
        grouped.setdefault(key, []).append((i, basis_str))

    # ---------------- Order within group ----------------
    ordered = {}
    for k, lst in grouped.items():
        if ordering == "reverse":
            lst.sort(key=lambda x: x[1], reverse=True)
        else:
            lst.sort(key=lambda x: x[1])
        ordered[k] = lst

    # ---------------- Build lattice ----------------
    group_keys = sorted(ordered.keys())
    rows, index_rows = [], []
    for k in group_keys:
        indices = [i for i, _ in ordered[k]]
        row = [psi[i] for i in indices]
        rows.append(row)
        index_rows.append(indices)

    return (ordered, rows, index_rows) if return_indices else (ordered, rows)
