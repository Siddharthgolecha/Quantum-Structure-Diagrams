import numpy as np


def compute_hld_metrics(psi, dims):
    """
    Compute global state-structure metrics for interpretation.
    """
    psi = np.asarray(psi, dtype=complex)
    norm = np.sum(np.abs(psi)**2)
    if not np.isclose(norm, 1.0):
        psi = psi / np.sqrt(norm)

    probs = np.abs(psi)**2
    amplitude_entropy = -(probs * np.log(probs + 1e-12))

    global_coherence = np.abs(np.sum(psi)) / len(psi)

    # Very simple proxy for entanglement visibility:
    # difference between highest and lowest amplitude magnitudes.
    entanglement_visibility = np.max(np.abs(psi)) - np.min(np.abs(psi))

    return {
        "global_coherence_spectrum": float(global_coherence),
        "entanglement_visibility_index": float(entanglement_visibility),
        "amplitude_entropy": {i: float(a) for i, a in enumerate(amplitude_entropy)},
    }
