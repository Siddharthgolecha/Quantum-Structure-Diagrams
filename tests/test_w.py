import numpy as np

from hld import compute_hld_metrics


def test_w_state_entropy_nonzero():
    psi = np.zeros(8, dtype=complex)
    psi[1] = psi[2] = psi[4] = 1/np.sqrt(3)
    m = compute_hld_metrics(psi, dims=[2,2,2])
    assert sum(m["amplitude_entropy"].values()) > 0
