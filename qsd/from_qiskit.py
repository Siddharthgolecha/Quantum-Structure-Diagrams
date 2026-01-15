import numpy as np


def state_from_statevector(statevector):
    """Accepts qiskit.quantum_info.Statevector or array."""
    try:
        return np.asarray(statevector.data, dtype=complex)
    except AttributeError:
        return np.asarray(statevector, dtype=complex)

def state_from_circuit(circuit, backend=None):
    """
    Execute circuit to obtain final statevector.
    If backend=None, uses Aer simulator if available.
    """
    try:
        from qiskit import Aer, execute
    except ImportError:
        raise ImportError("This feature requires qiskit: pip install qiskit")

    backend = backend or Aer.get_backend("statevector_simulator")
    job = execute(circuit, backend)
    result = job.result()
    return np.asarray(result.get_statevector(circuit), dtype=complex)
