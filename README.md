# HLD: Hilbert Lattice Diagram

[![PyPI version](https://badge.fury.io/py/hld.svg)](https://badge.fury.io/py/hld)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

The **Hilbert Lattice Diagram (HLD)** is a visualization and analysis framework for quantum states.
It organizes computational basis states into structured subspaces (rows) and orders states
systematically within them (columns). Each cell encodes **amplitude magnitude** (brightness) and
**phase** (hue), revealing coherence, correlations, and entanglement patterns directly from the
statevector.

---

## Overview

Quantum states often contain structure that is visually opaque in raw vector form. HLD provides a
compact and interpretable visual representation:

- **Brightness** → amplitude magnitude \\(|c_i|\\)
- **Hue (color)** → phase \\(\arg(c_i)\\)
- **Rows** → logical or excitation subspaces
- **Columns** → ordered basis states within each subspace

HLD supports **qubits and qudits** by specifying subsystem dimensions through `dims`.

---

## Installation

Install from PyPI:

```bash
pip install hld
```

Or clone and install from source:

```bash
git clone https://github.com/your-org/hld.git
cd hld
pip install -e .
```

---

## Getting Started

```python
from hld import plot_hld

# Example: Bell state
import numpy as np
psi = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)

plot_hld(psi, dims=[2, 2])
```

For combined analysis and visualization:

```python
from hld import analyze_and_plot_hld

metrics = analyze_and_plot_hld(psi, dims=[2,2], show_metrics=True)
```

---

## Integration with Qiskit (Optional)

```python
from qiskit import QuantumCircuit
from hld.from_qiskit import state_from_circuit
from hld import plot_hld

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)

psi = state_from_circuit(qc)
plot_hld(psi, dims=[2,2], show_metrics=True)
```

Install Qiskit if needed:

```bash
pip install qiskit
```

---

## Documentation

Full documentation (usage examples, interpretation notes, and metrics) is under development.

---

## Contributing

Contributions are welcome! Please open issues or pull requests.

---

## License

HLD is released under the **MIT License**.
