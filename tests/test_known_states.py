import numpy as np
import pytest
from qsd import compute_qsd_metrics


def test_psi_minus_metrics():
    # (|01> - |10>) / sqrt(2)
    psi = np.zeros(4, dtype=complex)
    psi[1] = 1 / np.sqrt(2)
    psi[2] = -1 / np.sqrt(2)
    m = compute_qsd_metrics(psi, dims=[2, 2])

    assert m["row_keys"] == ["1"]
    assert m["row_probabilities"]["1"] == pytest.approx(1.0)
    assert m["row_coherence"]["1"] == pytest.approx(0.0)
    assert m["row_amplitude_entropy"]["1"] == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.array(m["inter_row_phase_correlation"]), np.array([[1.0]])
    )
    np.testing.assert_allclose(np.array(m["inter_row_phase_offset"]), np.array([[0.0]]))
    assert m["global_coherence_index"] == pytest.approx(0.0)
    assert m["row_delocalization"] == pytest.approx(0.0)
    assert m["bipartite_entanglement_linear"]["value"] == pytest.approx(1.0)


def test_qutrit_bell_like_metrics():
    # (|00> + |22>) / sqrt(2)
    psi = np.zeros(9, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[8] = 1 / np.sqrt(2)
    m = compute_qsd_metrics(psi, dims=[3, 3])

    assert m["row_keys"] == ["0", "4"]
    assert m["row_probabilities"]["0"] == pytest.approx(0.5)
    assert m["row_probabilities"]["4"] == pytest.approx(0.5)
    assert m["row_coherence"]["0"] == pytest.approx(1.0)
    assert m["row_coherence"]["4"] == pytest.approx(1.0)
    assert m["row_amplitude_entropy"]["0"] == pytest.approx(0.0)
    assert m["row_amplitude_entropy"]["4"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        np.array(m["inter_row_phase_correlation"]), np.array([[1.0, 1.0], [1.0, 1.0]])
    )
    np.testing.assert_allclose(
        np.array(m["inter_row_phase_offset"]), np.array([[0.0, 0.0], [0.0, 0.0]])
    )
    assert m["global_coherence_index"] == pytest.approx(1.0)
    assert m["row_delocalization"] == pytest.approx(0.5)


def test_w_state_metrics():
    # (|001> + |010> + |100>) / sqrt(3)
    psi = np.zeros(8, dtype=complex)
    psi[1] = 1 / np.sqrt(3)
    psi[2] = 1 / np.sqrt(3)
    psi[4] = 1 / np.sqrt(3)
    m = compute_qsd_metrics(psi, dims=[2, 2, 2])

    assert m["row_keys"] == ["1"]
    assert m["row_probabilities"]["1"] == pytest.approx(1.0)
    assert m["row_coherence"]["1"] == pytest.approx(1.0)
    assert m["row_amplitude_entropy"]["1"] == pytest.approx(np.log2(3))
    np.testing.assert_allclose(
        np.array(m["inter_row_phase_correlation"]), np.array([[1.0]])
    )
    np.testing.assert_allclose(np.array(m["inter_row_phase_offset"]), np.array([[0.0]]))
    assert m["global_coherence_index"] == pytest.approx(1.0)
    assert m["row_delocalization"] == pytest.approx(0.0)
    assert m["bipartite_entanglement_linear"]["value"] == pytest.approx(8 / 9)


def test_haar_random_fixed_seed_metrics():
    np.random.seed(7)
    psi = np.random.randn(8) + 1j * np.random.randn(8)
    m = compute_qsd_metrics(psi, dims=[2, 2, 2])

    assert m["row_keys"] == ["0", "1", "2", "3"]
    assert sum(m["row_probabilities"].values()) == pytest.approx(1.0)
    assert m["row_delocalization"] == pytest.approx(0.639829728455874)
    assert m["global_coherence_index"] == pytest.approx(0.6788942677477805)
    assert m["bipartite_entanglement_linear"]["value"] == pytest.approx(0.9129115759240753)
