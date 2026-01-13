from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


def decode_basis_index(flat: int, dims: List[int]) -> Tuple[int, ...]:
    """Decode a flat basis index into mixed-radix levels given subsystem dims."""
    levels = []
    for d in reversed(dims):
        levels.append(flat % d)
        flat //= d
    return tuple(reversed(levels))


def _make_grouping_fn(
    dims: List[int],
    grouping: Union[str, Callable[[Tuple[int, ...]], Any]],
) -> Callable[[Tuple[int, ...]], Any]:
    """
    Returns a function G(levels)->row_label.

    grouping:
      - "auto": qubits -> hamming, otherwise -> excitation
      - "hamming": sum_k 1[level_k != 0]
      - "excitation": sum_k level_k
      - callable: user-provided grouping(levels)->label
    """
    if callable(grouping):
        return grouping

    if grouping == "auto":
        is_qubits = all(d == 2 for d in dims)
        grouping = "hamming" if is_qubits else "excitation"

    if grouping == "hamming":
        return lambda levels: int(sum(1 for x in levels if x != 0))

    if grouping == "excitation":
        return lambda levels: int(sum(levels))

    raise ValueError(f"Unknown grouping='{grouping}'. Use auto|hamming|excitation|callable.")


def _make_ordering_key(
    ordering: Union[str, Callable[[Tuple[int, ...], int], Any]]
) -> Callable[[Tuple[int, ...], int], Any]:
    """
    Returns a sort key function key(levels, flat_index)->something comparable.

    ordering:
      - "flat": by flat index
      - "lex": by levels tuple
      - callable: user-provided key(levels, flat_index)->key
    """
    if callable(ordering):
        return ordering

    if ordering == "flat":
        return lambda levels, flat: flat

    if ordering == "lex":
        return lambda levels, flat: levels

    raise ValueError(f"Unknown ordering='{ordering}'. Use flat|lex|callable.")


def bipartite_linear_entropy(
    psi: np.ndarray,
    dims: List[int],
    A_subsystems: List[int],
) -> Dict[str, float]:
    """
    Bipartite entanglement score for pure states via normalized linear entropy:
      E_lin(A:B) = (d_A/(d_A-1)) * (1 - Tr(rho_A^2))

    Returns dict with value, d_A, purity.
    """
    n = len(dims)
    A_subsystems = sorted(set(A_subsystems))
    if any(a < 0 or a >= n for a in A_subsystems):
        raise ValueError("A_subsystems contains invalid subsystem indices.")

    B_subsystems = [i for i in range(n) if i not in A_subsystems]
    if len(A_subsystems) == 0 or len(B_subsystems) == 0:
        # Trivial bipartition -> no entanglement
        return {"value": 0.0, "d_A": 1.0, "purity": 1.0}

    d_A = int(np.prod([dims[i] for i in A_subsystems]))
    d_B = int(np.prod([dims[i] for i in B_subsystems]))

    psi_t = psi.reshape(*dims)
    perm = A_subsystems + B_subsystems
    psi_perm = np.transpose(psi_t, perm)
    M = psi_perm.reshape(d_A, d_B)

    rho_A = M @ M.conj().T
    purity = float(np.real(np.trace(rho_A @ rho_A)))

    if d_A <= 1:
        return {"value": 0.0, "d_A": float(d_A), "purity": purity}

    E = (d_A / (d_A - 1.0)) * (1.0 - purity)
    # numerical safety
    E = float(np.clip(E, 0.0, 1.0))
    return {"value": E, "d_A": float(d_A), "purity": purity}


def compute_qsd_metrics(
    psi: Union[np.ndarray, List[complex]],
    dims: List[int],
    grouping: Union[str, Callable[[Tuple[int, ...]], Any]] = "auto",
    ordering: Union[str, Callable[[Tuple[int, ...], int], Any]] = "lex",
    A_subsystems: Optional[List[int]] = None,
    eps: float = 1e-15,
) -> Dict[str, Any]:
    """
    Compute finalized QSD-aligned metrics.

    Metrics returned:
      - row_keys: sorted populated row labels (as strings in dict keys)
      - row_sizes: N_r
      - row_probabilities: P_row(r)
      - row_coherence: C_row(r) = |sum c| / sum |c|
      - row_amplitude_entropy: S_amp(r) (Shannon base-2 over within-row probabilities)
      - inter_row_phase_correlation: Gamma_{r,s} = |<v_r,v_s>|/(||v_r|| ||v_s||)
      - inter_row_phase_offset: arg of normalized overlap (0 when undefined)
      - global_coherence_index: probability-weighted aggregate
      - row_delocalization: D_row = 1 - sum_r P_row(r)^2
      - bipartite_entanglement_linear: E_lin(A:B) (optional, defaults to first half)

    Notes:
      - Inter-row metrics depend on ordering (pi_r) because rows are compared columnwise.
      - Rows included match the QSD definition: only rows with nonzero population.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)

    N = int(np.prod(dims))
    if psi.size != N:
        raise ValueError(f"psi has length {psi.size}, but prod(dims)={N}.")

    # Normalize state
    norm2 = float(np.vdot(psi, psi).real)
    if norm2 <= 0:
        raise ValueError("State has zero norm.")
    psi = psi / np.sqrt(norm2)

    # Build grouping and ordering
    G = _make_grouping_fn(dims, grouping)
    key_fn = _make_ordering_key(ordering)

    # Decode all basis indices -> levels (vectorized when possible)
    levels = np.stack(np.unravel_index(np.arange(N), dims), axis=1)

    # Row labels (vectorized for built-ins; fallback to callable)
    if callable(grouping):
        row_labels = np.array([G(tuple(levels[i])) for i in range(N)], dtype=object)
    else:
        if grouping == "auto":
            is_qubits = all(d == 2 for d in dims)
            grouping = "hamming" if is_qubits else "excitation"
        if grouping == "hamming":
            row_labels = np.count_nonzero(levels, axis=1)
        elif grouping == "excitation":
            row_labels = np.sum(levels, axis=1)
        else:
            raise ValueError(f"Unknown grouping='{grouping}'. Use auto|hamming|excitation|callable.")

    # Collect indices by row label (only populated later)
    by_row: Dict[Any, List[int]] = {}

    # Determine ordering of basis indices within rows
    if callable(ordering):
        order = sorted(range(N), key=lambda i: key_fn(tuple(levels[i]), i))
    elif ordering == "flat":
        order = list(range(N))
    elif ordering == "lex":
        # Lexicographic order by levels tuple (first axis most significant)
        order = np.lexsort(levels[:, ::-1].T).tolist()
    else:
        raise ValueError(f"Unknown ordering='{ordering}'. Use flat|lex|callable.")

    for i in order:
        r = row_labels[i]
        by_row.setdefault(r, []).append(i)

    # Construct ordered row vectors for populated rows
    row_keys_all = sorted(by_row.keys())
    row_vecs = []
    row_keys_pop = []
    row_sizes = {}
    row_probs = {}
    row_coh = {}
    row_ent = {}

    abs_psi = np.abs(psi)
    abs2_psi = abs_psi ** 2

    for r in row_keys_all:
        idxs = by_row[r]
        vec = psi[idxs]

        P = float(np.sum(abs2_psi[idxs]))
        if P <= eps:
            # QSD rows are defined as those with nonzero amplitude support
            continue

        # Row coherence: |sum c| / sum |c|
        denom = float(np.sum(abs_psi[idxs]))
        C = float(np.abs(np.sum(vec)) / denom) if denom > eps else float("nan")

        # Row amplitude entropy (Shannon base-2) over within-row probs
        pr = abs2_psi[idxs] / P
        # safe log: ignore zeros
        pr_nz = pr[pr > 0]
        S = float(-np.sum(pr_nz * np.log2(pr_nz)))

        row_keys_pop.append(r)
        row_vecs.append(vec)
        row_sizes[str(r)] = int(vec.size)
        row_probs[str(r)] = P
        row_coh[str(r)] = C
        row_ent[str(r)] = S

    if len(row_keys_pop) == 0:
        raise ValueError("No populated rows found (unexpected after normalization).")

    # Convert ragged row vectors into padded matrix V (W x K)
    W = len(row_keys_pop)
    K = max(v.size for v in row_vecs)
    V = np.zeros((W, K), dtype=complex)
    for i, v in enumerate(row_vecs):
        V[i, : v.size] = v

    # Inter-row normalized overlap
    Gmat = V @ V.conj().T  # complex Gram matrix
    norms = np.sqrt(np.real(np.diag(Gmat)))
    denom = norms[:, None] * norms[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(denom > eps, Gmat / denom, 0.0 + 0.0j)

    Gamma_mag = np.abs(gamma).astype(float)  # in [0,1] up to numerical error
    Gamma_phase = np.angle(gamma).astype(float)

    # Global coherence index (probability-weighted aggregate)
    P = np.array([row_probs[str(r)] for r in row_keys_pop], dtype=float)
    C = np.array([row_coh[str(r)] for r in row_keys_pop], dtype=float)

    # weights w_rs = P_r P_s, exclude diagonal
    Wmat = np.outer(P, P)
    np.fill_diagonal(Wmat, 0.0)
    denom_w = float(np.sum(Wmat))

    if denom_w <= eps:
        # only one populated row effectively
        C_global = float(C[0])
    else:
        num = float(np.sum(Wmat * (C[:, None] * C[None, :]) * Gamma_mag))
        C_global = num / denom_w

    # Row delocalization
    D_row = float(1.0 - np.sum(P ** 2))
    D_row = float(np.clip(D_row, 0.0, 1.0))

    # Bipartite entanglement score (optional but recommended)
    if A_subsystems is None:
        # default: first half vs rest
        n = len(dims)
        A_subsystems = list(range(max(1, n // 2)))
    ent = bipartite_linear_entropy(psi, dims, A_subsystems)

    return {
        "meta": {
            "dims": list(map(int, dims)),
            "grouping": grouping if isinstance(grouping, str) else "callable",
            "ordering": ordering if isinstance(ordering, str) else "callable",
            "A_subsystems": list(map(int, A_subsystems)),
        },
        "row_keys": [str(r) for r in row_keys_pop],
        "row_sizes": row_sizes,
        "row_probabilities": row_probs,
        "row_coherence": row_coh,
        "row_amplitude_entropy": row_ent,
        "inter_row_phase_correlation": Gamma_mag.tolist(),  # corresponds to Gamma_{r,s}
        "inter_row_phase_offset": Gamma_phase.tolist(),     # arg(gamma_{r,s})
        "global_coherence_index": float(C_global),
        "row_delocalization": float(D_row),
        "bipartite_entanglement_linear": ent,
    }
