import numpy as np

from hld import analyze_and_plot_hld

# GHZ (3 qubits)
psi_ghz = np.zeros(8, dtype=complex); psi_ghz[0]=psi_ghz[7]=1/np.sqrt(2)
analyze_and_plot_hld(psi_ghz, dims=[2,2,2], theme="light", show_metrics=True,
                     save_path="./figures/ghz_hld_light.png",
                     caption="HLD of GHZ; hue=phase, brightness=|amp|.",
                     style="paper")

# W (3 qubits)
psi_w = np.zeros(8, dtype=complex); psi_w[1]=psi_w[2]=psi_w[4]=1/np.sqrt(3)
analyze_and_plot_hld(psi_w, dims=[2,2,2], theme="light", show_metrics=True,
                     save_path="./figures/w_hld_light.png",
                     caption="HLD of W state; single-excitation layer visualized.",
                     style="paper")

# Qutrit Bell-like (2 qutrits)
psi_qut = np.zeros(9, dtype=complex); psi_qut[0]=psi_qut[8]=1/np.sqrt(2)
analyze_and_plot_hld(psi_qut, dims=[3,3], theme="light", show_metrics=True,
                     save_path="./figures/qutrit_bell_hld_light.png",
                     caption="HLD of two-qutrit Bell-like state.",
                     style="paper")

# Haar-random 3-qubit
np.random.seed(7)
psi_rand = np.random.randn(8)+1j*np.random.randn(8); psi_rand/=np.linalg.norm(psi_rand)
analyze_and_plot_hld(psi_rand, dims=[2,2,2], theme="light", show_metrics=True,
                     save_path="./figures/haar3_hld_light.png",
                     caption="Haar-random 3-qubit state.",
                     style="paper")
