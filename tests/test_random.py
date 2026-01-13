import numpy as np
from qsd import compute_qsd_metrics


def test_metrics_output_types():
    psi = np.random.randn(8) + 1j * np.random.randn(8)
    m = compute_qsd_metrics(psi, dims=[2, 2, 2])

    assert isinstance(m, dict)
    assert isinstance(m["meta"], dict)
    assert isinstance(m["row_keys"], list)
    assert isinstance(m["row_sizes"], dict)
    assert isinstance(m["row_probabilities"], dict)
    assert isinstance(m["row_coherence"], dict)
    assert isinstance(m["row_amplitude_entropy"], dict)
    assert isinstance(m["inter_row_phase_correlation"], list)
    assert isinstance(m["inter_row_phase_offset"], list)
    assert isinstance(m["global_coherence_index"], float)
    assert isinstance(m["row_delocalization"], float)
    assert isinstance(m["bipartite_entanglement_linear"], dict)
