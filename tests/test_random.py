import numpy as np
from qsd import compute_qsd_metrics


def test_metrics_output_types():
    psi = np.random.randn(8) + 1j * np.random.randn(8)
    m = compute_qsd_metrics(psi)

    assert isinstance(m, dict)
    assert isinstance(m["meta"], dict)
    assert isinstance(m["row_keys"], list)
    assert isinstance(m["row_sizes"], dict)
    assert isinstance(m["row_probabilities"], dict)
    assert isinstance(m["row_amplitude_entropy"], dict)
    assert isinstance(m["row_phase_alignment"], dict)
    assert isinstance(m["row_effective_support"], dict)
    assert isinstance(m["row_collision_entropy"], dict)
    assert isinstance(m["row_participation_ratio"], dict)
    assert isinstance(m["row_nonzero_count"], dict)
    assert isinstance(m["row_delocalization"], float)
    assert isinstance(m["bipartite_entanglement_linear"], dict)
    assert isinstance(m["aligned_separability_test"], dict)
    assert isinstance(m["row_partition_sizes_all"], dict)
    assert isinstance(m["haar_baselines"], dict)
