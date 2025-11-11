
# HLD Demonstration Notebook
# ==========================
# This notebook demonstrates visualization and analysis of quantum states using the Hilbert Lattice Diagram (HLD).
# It plots GHZ, W, and a random 3-qubit state in both dark and light themes, and prints their computed metrics.

import numpy as np

from hld import analyze_and_plot_hld, compute_hld_metrics

# GHZ state |000> + |111>
psi_ghz = np.zeros(8, dtype=complex)
psi_ghz[0] = psi_ghz[7] = 1/np.sqrt(2)

# W state (|001> + |010> + |100>)/sqrt(3)
psi_w = np.zeros(8, dtype=complex)
psi_w[1] = psi_w[2] = psi_w[4] = 1/np.sqrt(3)

# Random normalized 3-qubit state
np.random.seed(42)
psi_rand = np.random.randn(8) + 1j*np.random.randn(8)
psi_rand /= np.linalg.norm(psi_rand)

# Plot GHZ (dark)
analyze_and_plot_hld(psi_ghz, dims=[2,2,2], theme="dark", show_metrics=True, save_path="ghz_hld_dark.png")

# Plot W (light)
analyze_and_plot_hld(psi_w, dims=[2,2,2], theme="light", show_metrics=True, save_path="w_hld_light.png")

# Plot random (dark)
analyze_and_plot_hld(psi_rand, dims=[2,2,2], theme="dark", show_metrics=True, save_path="random_hld_dark.png")

# Print metrics summary
for name, psi in [("GHZ", psi_ghz), ("W", psi_w), ("Random", psi_rand)]:
    m = compute_hld_metrics(psi, dims=[2,2,2])
    print(f"\\n{name} state metrics:")
    print(f"  Global Coherence: {m['global_coherence_spectrum']:.3f}")
    print(f"  Entanglement Visibility: {m['entanglement_visibility_index']:.3f}")
    mean_S = float(np.mean(list(m['amplitude_entropy'].values())))
    print(f"  Mean Amplitude Entropy: {mean_S:.3f}")
