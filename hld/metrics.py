import numpy as np


def decode_basis_index(flat, dims):
    levels = []
    for d in reversed(dims):
        levels.append(flat % d)
        flat //= d
    return tuple(reversed(levels))

def compute_hld_metrics(psi, dims):
    """
    Enhanced HLD metrics with safe handling for empty rows.
    Works for qubits & qudits. Row label = sum(levels) by default.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    # normalize
    p = np.abs(psi)**2
    s = p.sum()
    if s == 0:
        raise ValueError("State has zero norm.")
    psi = psi / np.sqrt(s)
    p = np.abs(psi)**2  # recompute after norm

    # ---- Row labels (sum of digit-levels) ----
    idx_levels = [decode_basis_index(i, dims) for i in range(len(psi))]
    row_labels = np.array([sum(t) for t in idx_levels], dtype=int)
    n_rows = int(row_labels.max()) + 1 if row_labels.size else 0

    # ---- Row probabilities (fast via bincount) ----
    row_probs = np.bincount(row_labels, weights=p, minlength=n_rows).astype(float)
    row_probabilities = {int(r): float(val) for r, val in enumerate(row_probs)}

    # ---- Intra-row coherence (safe divide) ----
    intra_row_coherence = {}
    for r in range(n_rows):
        idxs = np.where(row_labels == r)[0]
        if idxs.size == 0:
            intra_row_coherence[r] = np.nan   # no elements in this row
            continue
        vec = psi[idxs]
        denom = np.sum(np.abs(vec))
        if denom == 0:
            intra_row_coherence[r] = np.nan   # empty-mass row
        else:
            intra_row_coherence[r] = float(np.abs(np.sum(vec)) / denom)

    # ---- Inter-row correlation (cosine-like, safe) ----
    inter_row_correlation_matrix = np.zeros((n_rows, n_rows), dtype=float)
    # Row coherence vectors (sums, not full projections)
    row_sums = np.zeros(n_rows, dtype=complex)
    for r in range(n_rows):
        row_sums[r] = np.sum(psi[row_labels == r])

    inter_row_correlation_matrix = np.zeros((n_rows, n_rows), dtype=float)
    for i in range(n_rows):
        for j in range(n_rows):
            inter_row_correlation_matrix[i, j] = float(np.abs(row_sums[i] * np.conj(row_sums[j])))

    total = np.sum(np.abs(row_sums))
    if total > 0:
        inter_row_correlation_matrix = inter_row_correlation_matrix / (total**2)
    inter_row_correlation_matrix = inter_row_correlation_matrix.tolist()

    # ---- Amplitude entropy per basis index ----
    # small epsilon for numerical stability
    eps = 1e-12
    amp_entropy = -(p * np.log(p + eps))
    amplitude_entropy = {int(i): float(a) for i, a in enumerate(amp_entropy)}

    # ---- Global coherence (your original) ----
    global_coherence_spectrum = float(np.abs(np.sum(psi)) / len(psi))

    # ---- Entanglement visibility (your proxy) ----
    entanglement_visibility_index = float(np.max(np.abs(psi)) - np.min(np.abs(psi)))

    return {
        "row_probabilities": row_probabilities,
        "intra_row_coherence": intra_row_coherence,  # may include NaNs for empty rows
        "inter_row_correlation_matrix": inter_row_correlation_matrix,
        "amplitude_entropy": amplitude_entropy,
        "global_coherence_spectrum": global_coherence_spectrum,
        "entanglement_visibility_index": entanglement_visibility_index,
    }
