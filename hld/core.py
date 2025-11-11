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


def build_hld_lattice(psi, dims, grouping="hamming", ordering="lex"):
    """
    Convert statevector into a ragged 2D array of complex amplitudes.
    """
    psi = np.asarray(psi, dtype=complex)
    grouped = group_basis_states(dims, grouping)
    ordered = order_basis_states(grouped, ordering)

    rows = []
    for r, items in ordered.items():
        row = [psi[i] for (i, _) in items]
        rows.append(row)

    return ordered, rows
