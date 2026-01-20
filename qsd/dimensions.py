from typing import Iterable, List

import numpy as np


def _infer_dims_from_length(n_states: int) -> List[int]:
    """Infer subsystem dimensions from a statevector length.

    Heuristic: prefer the factorization with the largest number of equal-sized
    subsystems; otherwise return [n_states] for a single subsystem.

    Example:
        >>> _infer_dims_from_length(8)
        [2, 2, 2]
    """
    if n_states <= 0:
        raise ValueError("Statevector length must be positive.")

    # No hints: choose the factorization with the largest number of qudits.
    best = None  # (m, base)
    max_base = int(np.sqrt(n_states))
    for base in range(2, max_base + 1):
        remaining = n_states
        m = 0
        while remaining % base == 0:
            remaining //= base
            m += 1
        if remaining == 1 and m > 1:
            if best is None or m > best[0]:
                best = (m, base)

    if best is not None:
        m, base = best
        return [base] * m

    return [n_states]


def resolve_dims(
    psi: Iterable[complex],
) -> List[int]:
    """Resolve dimensions for a statevector-like iterable.

    Args:
        psi: Iterable of complex amplitudes.

    Returns:
        List of inferred dimensions.

    Example:
        >>> resolve_dims([1, 0, 0, 0])
        [2, 2]
    """
    psi_arr = np.asarray(psi, dtype=complex).reshape(-1)
    return _infer_dims_from_length(int(psi_arr.size))
