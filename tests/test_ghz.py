import numpy as np
import pytest
from qsd import compute_qsd_metrics


def test_ghz_metrics():
    psi = np.zeros(8, dtype=complex)
    psi[0] = psi[7] = 1 / np.sqrt(2)
    m = compute_qsd_metrics(psi)

    assert m["row_keys"] == ["0", "3"]
    assert m["row_probabilities"]["0"] == pytest.approx(0.5)
    assert m["row_probabilities"]["3"] == pytest.approx(0.5)
    assert m["row_phase_alignment"]["0"] == pytest.approx(1.0)
    assert m["row_phase_alignment"]["3"] == pytest.approx(1.0)
    assert m["row_amplitude_entropy"]["0"] == pytest.approx(0.0)
    assert m["row_amplitude_entropy"]["3"] == pytest.approx(0.0)
    assert m["row_effective_support"]["0"] == pytest.approx(1.0)
    assert m["row_effective_support"]["3"] == pytest.approx(1.0)
    assert m["row_delocalization"] == pytest.approx(0.5)
    assert m["bipartite_entanglement_linear"]["value"] == pytest.approx(1.0)
