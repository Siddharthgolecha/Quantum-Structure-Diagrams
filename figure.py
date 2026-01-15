import json

import numpy as np

from qsd import analyze_and_plot_qsd


def run_example(name, psi, dims, *, save_path, caption, grouping=None, ordering=None):
    """Run QSD analysis, plot the figure, and persist metrics to disk."""
    plot_kwargs = {
        "dims": dims,
        "theme": "light",
        "show_metrics": True,
        "save_path": save_path,
        "caption": caption,
        "style": "paper",
    }
    if grouping is not None:
        plot_kwargs["grouping"] = grouping
    if ordering is not None:
        plot_kwargs["ordering"] = ordering

    metrics = analyze_and_plot_qsd(psi, **plot_kwargs)

    metrics_path = f"./figures/{name}_metrics.json"
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print(f"{name} metrics:")
    print(json.dumps(metrics, indent=2, sort_keys=True))

# GHZ (3 qubits): (|000> + |111>) / sqrt(2)
psi_ghz = np.zeros(8, dtype=complex)
psi_ghz[0] = psi_ghz[7] = 1 / np.sqrt(2)
run_example(
    "ghz_qsd_light",
    psi_ghz,
    dims=[2, 2, 2],
    save_path="./figures/ghz_qsd_light.png",
    caption="QSD of GHZ; hue=phase, brightness=|amp|.",
)

# Psi-minus Bell state (2 qubits): (|01> - |10>) / sqrt(2)
psi_minus = np.zeros(4, dtype=complex)
psi_minus[1] = 1 / np.sqrt(2)
psi_minus[2] = -1 / np.sqrt(2)
run_example(
    "psi_minus_bell_qsd_light",
    psi_minus,
    dims=[2, 2],
    save_path="./figures/psi_minus_bell_qsd_light.png",
    caption=r"QSD of Bell state $|\Psi-\rangle$; hue=phase, brightness=|amp|.",
)

# W (3 qubits): (|001> + |010> + |100>) / sqrt(3)
psi_w = np.zeros(8, dtype=complex)
psi_w[1] =  psi_w[2] = psi_w[4] = 1 / np.sqrt(3)
run_example(
    "w_qsd_light",
    psi_w,
    dims=[2, 2, 2],
    save_path="./figures/w_qsd_light.png",
    caption="QSD of W state; single-excitation layer visualized.",
)

# Qutrit Bell-like (2 qutrits): (|00> + |22>) / sqrt(2)
psi_qut = np.zeros(9, dtype=complex)
psi_qut[0] = psi_qut[8] = 1 / np.sqrt(2)
run_example(
    "qutrit_bell_qsd_light",
    psi_qut,
    dims=[3, 3],
    save_path="./figures/qutrit_bell_qsd_light.png",
    caption="QSD of two-qutrit Bell-like state (row = sum of levels).",
    grouping="levelsum",
)

# Haar-random 3-qubit example (fixed seed for reproducibility)
np.random.seed(7)
psi_rand = np.random.randn(8) + 1j * np.random.randn(8)
psi_rand /= np.linalg.norm(psi_rand)
run_example(
    "haar3_qsd_light",
    psi_rand,
    dims=[2, 2, 2],
    save_path="./figures/haar3_qsd_light.png",
    caption="Haar-random 3-qubit state.",
)
