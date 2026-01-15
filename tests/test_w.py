import numpy as np
from qsd import compute_qsd_metrics


def test_w_state_entropy_nonzero():
    psi = np.zeros(8, dtype=complex)
    psi[1] = psi[2] = psi[4] = 1/np.sqrt(3)
    m = compute_qsd_metrics(psi)
    assert sum(m["row_amplitude_entropy"].values()) > 0
