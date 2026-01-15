from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .dimensions import resolve_dims

def decode_basis_index(flat: int, dims: List[int]) -> Tuple[int, ...]:
    """Decode a flat basis index into mixed-radix levels given subsystem dims."""
    levels = []
    for d in reversed(dims):
        levels.append(flat % d)
        flat //= d
    return tuple(reversed(levels))


def _row_label_sort_key(label: Any) -> Tuple[int, Any]:
    """Robust ordering for row labels (ints first, then floats, then strings)."""
    if isinstance(label, (int, np.integer)):
        return (0, int(label))
    if isinstance(label, (float, np.floating)):
        return (1, float(label))
    return (2, str(label))


def _resolve_grouping(
    dims: List[int], grouping: Union[str, Callable]
) -> Tuple[str, Callable]:
    """
    Resolve grouping into (grouping_resolved_name, G(levels)->row_label).
    """
    if callable(grouping):
        return ("callable", grouping)

    grouping_requested = grouping
    if grouping_requested == "auto":
        grouping_requested = "excitation"

    if grouping_requested in ("hamming", "excitation"):
        # LaTeX definition: nonzero count
        return ("excitation", lambda levels: int(sum(1 for x in levels if x != 0)))
    if grouping_requested == "parity":
        return ("parity", lambda levels: int(sum(levels) % 2))
    if grouping_requested in ("flat", "none"):
        return ("flat", lambda levels: 0)
    if grouping_requested in ("levelsum", "sumlevels"):
        return ("levelsum", lambda levels: int(sum(levels)))

    raise ValueError(
        f"Unknown grouping='{grouping}'. "
        "Use auto|excitation|hamming|parity|flat|none|levelsum|callable."
    )


def _harmonic_numbers_up_to(m: int) -> np.ndarray:
    """H_k for k=1..m as float array of length m."""
    if m <= 0:
        return np.array([], dtype=float)
    return np.cumsum(1.0 / np.arange(1, m + 1, dtype=float))


def bipartite_linear_entropy(
    psi: np.ndarray,
    A_subsystems: List[int],
) -> Dict[str, float]:
    """
    Pure-state bipartite entanglement via normalized linear entropy:

        E_lin(A:B) = (d_A/(d_A-1)) * (1 - Tr(rho_A^2))

    Returns: {"value": E, "d_A": d_A, "purity": Tr(rho_A^2)}
    """
    dims = resolve_dims(psi)
    n = len(dims)
    A_subsystems = sorted(set(A_subsystems))
    if any(a < 0 or a >= n for a in A_subsystems):
        raise ValueError("A_subsystems contains invalid subsystem indices.")

    B_subsystems = [i for i in range(n) if i not in A_subsystems]
    if len(A_subsystems) == 0 or len(B_subsystems) == 0:
        return {"value": 0.0, "d_A": 1.0, "purity": 1.0}

    d_A = int(np.prod([dims[i] for i in A_subsystems]))
    d_B = int(np.prod([dims[i] for i in B_subsystems]))

    psi_t = psi.reshape(*dims)
    perm = A_subsystems + B_subsystems
    psi_perm = np.transpose(psi_t, perm)
    M = psi_perm.reshape(d_A, d_B)

    rho_A = M @ M.conj().T
    purity = float(np.trace(rho_A @ rho_A).real)

    if d_A <= 1:
        return {"value": 0.0, "d_A": float(d_A), "purity": purity}

    E = (d_A / (d_A - 1.0)) * (1.0 - purity)
    E = float(np.clip(E, 0.0, 1.0))
    return {"value": E, "d_A": float(d_A), "purity": purity}


def bipartite_rank_one_test(
    psi: np.ndarray,
    A_subsystems: List[int],
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Numerical companion to Prop. 'Row colinearity and separability' in the bipartition-aligned case.

    Returns a robust rank-1 test for the coefficient matrix M (Schmidt rank 1 ↔ separable),
    implemented via singular values.
    """
    dims = resolve_dims(psi)
    n = len(dims)
    A_subsystems = sorted(set(A_subsystems))
    if any(a < 0 or a >= n for a in A_subsystems):
        raise ValueError("A_subsystems contains invalid subsystem indices.")
    B_subsystems = [i for i in range(n) if i not in A_subsystems]
    if len(A_subsystems) == 0 or len(B_subsystems) == 0:
        return {
            "rank_one": True,
            "sigma_ratio": 0.0,
            "d_A": 1,
            "d_B": int(np.prod(dims)),
        }

    d_A = int(np.prod([dims[i] for i in A_subsystems]))
    d_B = int(np.prod([dims[i] for i in B_subsystems]))

    psi_t = psi.reshape(*dims)
    perm = A_subsystems + B_subsystems
    M = np.transpose(psi_t, perm).reshape(d_A, d_B)

    s = np.linalg.svd(M, compute_uv=False)
    if s.size <= 1:
        return {"rank_one": True, "sigma_ratio": 0.0, "d_A": d_A, "d_B": d_B}

    sigma_ratio = float(s[1] / (s[0] + 0.0))
    rank_one = bool(sigma_ratio <= tol)
    return {"rank_one": rank_one, "sigma_ratio": sigma_ratio, "d_A": d_A, "d_B": d_B}


def compute_qsd_metrics(
    psi: Union[np.ndarray, List[complex]],
    grouping: Union[str, Callable[[Tuple[int, ...]], Any]] = "auto",
    ordering: Union[str, Callable[[Tuple[int, ...], int], Any]] = "lex",
    A_subsystems: Optional[List[int]] = None,
    eps: float = 1e-15,
) -> Dict[str, Any]:
    """
    Compute QSD-aligned metrics.

    Local dimensions are inferred from the statevector length (assumes equal dims).

    Returns (populated rows only, i.e., P_row(r) > eps):
      - row_sizes:              N_r
      - row_probabilities:      P_row(r)
      - row_amplitude_entropy:  S_amp(r)  (base-2, conditional within-row)
      - row_effective_support:  N_eff(r) = 2^{S_amp(r)}
      - row_collision_entropy:  H2(r) = -log2(sum p^2)
      - row_participation_ratio:PR(r) = 1/sum p^2
      - row_nonzero_count:      k_r  (# entries with |c|^2 > eps inside the row)
      - row_phase_alignment:    C_row(r) = |sum c| / sum |c|
      - row_delocalization:     D_row = 1 - sum_r P_row(r)^2
      - bipartite_entanglement_linear: E_lin(A:B)  (reference metric)
      - aligned_separability_test: rank-one test via SVD (numerical companion)

    Also returns Haar baselines (Prop. Haar baselines...) for the *partition* induced by G:
      - haar_baselines: for each row label r in the partition:
            E[P_row(r)], Var(P_row(r)), E[S_amp(r)], E[sum p^2]
    """
    dims = resolve_dims(psi)
    psi = np.asarray(psi, dtype=complex).reshape(-1)

    N = int(np.prod(dims))
    if psi.size != N:
        raise ValueError(f"psi has length {psi.size}, but prod(dims)={N}.")

    # Normalize state
    norm2 = float(np.vdot(psi, psi).real)
    if norm2 <= 0:
        raise ValueError("State has zero norm.")
    psi = psi / np.sqrt(norm2)

    grouping_resolved, G = _resolve_grouping(dims, grouping)

    # ordering is accepted for API symmetry, but not used by scalar metrics
    ordering_resolved = "ignored"
    if isinstance(ordering, dict):
        ordering_resolved = "per-row"
    elif callable(ordering):
        ordering_resolved = "callable"
    elif isinstance(ordering, str):
        ordering_resolved = ordering

    # Decode all basis indices -> levels (shape: N x nsub)
    levels = np.stack(np.unravel_index(np.arange(N), dims), axis=1)

    if callable(grouping):
        row_labels = np.array([G(tuple(levels[i])) for i in range(N)], dtype=object)
    else:
        if grouping_resolved == "excitation":
            row_labels = np.count_nonzero(levels, axis=1)
        elif grouping_resolved == "parity":
            row_labels = np.sum(levels, axis=1) % 2
        elif grouping_resolved == "flat":
            row_labels = np.zeros(N, dtype=int)
        elif grouping_resolved == "levelsum":
            row_labels = np.sum(levels, axis=1)
        else:
            row_labels = np.array([G(tuple(levels[i])) for i in range(N)], dtype=object)

    by_row: Dict[Any, List[int]] = {}
    for i in range(N):
        r = row_labels[i]
        by_row.setdefault(r, []).append(i)

    row_keys_all = sorted(by_row.keys(), key=_row_label_sort_key)

    # Compute row labels
    if callable(grouping):
        row_labels = np.array([G(tuple(levels[i])) for i in range(N)], dtype=object)
    else:
        # vectorize built-ins for speed/clarity
        if grouping_resolved == "hamming":
            row_labels = np.count_nonzero(levels, axis=1)
        elif grouping_resolved == "excitation":
            row_labels = np.sum(levels, axis=1)
        else:
            # should never happen due to _resolve_grouping
            row_labels = np.array([G(tuple(levels[i])) for i in range(N)], dtype=object)

    # Ordering does not affect these scalar metrics; ignore it entirely.
    by_row: Dict[Any, List[int]] = {}
    for i in range(N):
        r = row_labels[i]
        by_row.setdefault(r, []).append(i)

    # Partition info (all rows)
    row_keys_all = sorted(by_row.keys(), key=_row_label_sort_key)
    row_sizes_all = {str(r): int(len(by_row[r])) for r in row_keys_all}

    # Precompute amplitude magnitudes
    abs_psi = np.abs(psi)
    abs2_psi = abs_psi**2

    # Row metrics (populated rows only)
    row_keys_pop: List[Any] = []
    row_sizes: Dict[str, int] = {}
    row_probs: Dict[str, float] = {}

    row_Samp: Dict[str, float] = {}
    row_Neff: Dict[str, float] = {}
    row_H2: Dict[str, float] = {}
    row_PR: Dict[str, float] = {}
    row_k: Dict[str, int] = {}

    row_Crow: Dict[str, float] = {}

    for r in row_keys_all:
        idxs = by_row[r]
        P = float(np.sum(abs2_psi[idxs]))
        if P <= eps:
            continue

        key = str(r)
        row_keys_pop.append(r)
        row_sizes[key] = int(len(idxs))
        row_probs[key] = P

        # k_r = number of entries with |c|^2 > eps
        k_r = int(np.count_nonzero(abs2_psi[idxs] > eps))
        row_k[key] = k_r

        # Conditional within-row probabilities
        pr = abs2_psi[idxs] / P
        pr_nz = pr[pr > 0]

        # S_amp(r)
        S = float(-np.sum(pr_nz * np.log2(pr_nz))) if pr_nz.size else 0.0
        row_Samp[key] = S
        row_Neff[key] = float(2.0**S)

        # Collision entropy H2 and participation ratio PR
        sum_p2 = float(np.sum(pr * pr))
        if sum_p2 > 0:
            row_H2[key] = float(-np.log2(sum_p2))
            row_PR[key] = float(1.0 / sum_p2)
        else:
            row_H2[key] = float("nan")
            row_PR[key] = float("nan")

        # C_row(r) = |sum c| / sum |c|
        denom = float(np.sum(abs_psi[idxs]))
        if denom > eps:
            row_Crow[key] = float(np.abs(np.sum(psi[idxs])) / denom)
        else:
            row_Crow[key] = float("nan")

    if not row_keys_pop:
        raise ValueError("No populated rows found (unexpected after normalization).")

    # Row delocalization D_row = 1 - sum_r P_row(r)^2  (over populated rows; zeros don't matter)
    P_vec = np.array([row_probs[str(r)] for r in row_keys_pop], dtype=float)
    D_row = float(1.0 - np.sum(P_vec**2))
    D_row = float(np.clip(D_row, 0.0, 1.0))

    # Default bipartition for reference entanglement
    if A_subsystems is None:
        n = len(dims)
        A_subsystems = list(range(max(1, n // 2)))

    ent = bipartite_linear_entropy(psi, A_subsystems)
    sep_test = bipartite_rank_one_test(psi, A_subsystems)

    # Haar baselines for the partition induced by G (Prop. Haar baselines...)
    max_Nr = max(row_sizes_all.values()) if row_sizes_all else 0
    H = _harmonic_numbers_up_to(max_Nr)
    ln2 = float(np.log(2.0))

    haar: Dict[str, Dict[str, float]] = {}
    for r in row_keys_all:
        Nr = row_sizes_all[str(r)]
        mean_P = float(Nr / N)
        var_P = float(Nr * (N - Nr) / (N * N * (N + 1.0))) if N > 0 else 0.0
        # E[S_amp] = (H_Nr - 1)/ln 2  with H_Nr harmonic number
        ES = float((H[Nr - 1] - 1.0) / ln2) if Nr >= 1 else 0.0
        # E[sum p^2] = 2/(Nr+1) under Dirichlet(1,...,1)
        E_sum_p2 = float(2.0 / (Nr + 1.0)) if Nr >= 1 else float("nan")

        haar[str(r)] = {
            "N_r": float(Nr),
            "E_P_row": mean_P,
            "Var_P_row": var_P,
            "E_S_amp": ES,
            "E_sum_p2": E_sum_p2,
        }

    return {
        "meta": {
            "dims": list(map(int, dims)),
            "grouping_requested": grouping if isinstance(grouping, str) else "callable",
            "grouping_resolved": grouping_resolved,
            "ordering_requested": (
                ordering
                if isinstance(ordering, str)
                else ("per-row" if isinstance(ordering, dict) else "callable")
            ),
            "ordering_resolved": ordering_resolved,
            "A_subsystems": list(map(int, A_subsystems)),
            "eps": float(eps),
        },
        "row_keys": [str(r) for r in sorted(row_keys_pop, key=_row_label_sort_key)],
        "row_sizes": row_sizes,
        "row_probabilities": row_probs,
        "row_amplitude_entropy": row_Samp,  # S_amp(r)
        "row_effective_support": row_Neff,  # 2^{S_amp(r)}
        "row_collision_entropy": row_H2,  # H2(r)
        "row_participation_ratio": row_PR,  # PR(r)
        "row_nonzero_count": row_k,  # k_r
        "row_phase_alignment": row_Crow,  # C_row(r)
        "row_delocalization": float(D_row),  # D_row
        "bipartite_entanglement_linear": ent,  # E_lin(A:B)
        "aligned_separability_test": sep_test,  # rank-one test (numerical)
        "row_partition_sizes_all": row_sizes_all,  # sizes for *all* rows in the partition
        "haar_baselines": haar,  # Prop. Haar baselines
    }
