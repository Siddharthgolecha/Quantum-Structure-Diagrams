import numpy as np

from hld import compute_hld_metrics


def test_random_state_normalized():
    psi = np.random.randn(8) + 1j*np.random.randn(8)
    m = compute_hld_metrics(psi, dims=[2,2,2])
    assert "global_coherence_spectrum" in m
