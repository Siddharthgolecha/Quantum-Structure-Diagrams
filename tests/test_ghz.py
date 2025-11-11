import numpy as np

from hld import compute_hld_metrics


def test_ghz_metrics():
    psi = np.zeros(8, dtype=complex)
    psi[0] = psi[7] = 1/np.sqrt(2)
    m = compute_hld_metrics(psi, dims=[2,2,2])
    assert m["global_coherence_spectrum"] > 0.4
