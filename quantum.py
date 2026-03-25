"""Grover's algorithm components: oracle, diffuser, and search runner."""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer


def grover_oracle(qc, marked_state):
    """Apply the Grover oracle for a given marked state bit-string."""
    n = len(marked_state)
    for i, bit in enumerate(marked_state):
        if bit == "0":
            qc.x(i)
    qc.h(n - 1)
    if n > 1:
        qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for i, bit in enumerate(marked_state):
        if bit == "0":
            qc.x(i)


def diffuser(qc, n):
    """Apply the Grover diffusion operator."""
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    if n > 1:
        qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


def run_grover_search(n_candidates, best_classical, shots=1024):
    """
    Build and execute a Grover circuit targeting the best classical candidate.

    Returns
    -------
    grover_index : int
        The index amplified by Grover's algorithm.
    counts : dict
        Raw measurement counts.
    qc : QuantumCircuit
        The circuit that was executed.
    n_qubits : int
    marked_state : str
    iterations : int
    """
    n_qubits = max(1, int(np.ceil(np.log2(n_candidates))))
    marked_state = format(best_classical, f"0{n_qubits}b")

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    # Use actual candidate count so iterations reflect each run's search space.
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(n_candidates))))
    for _ in range(iterations):
        grover_oracle(qc, marked_state)
        diffuser(qc, n_qubits)
    qc.measure_all()

    backend = Aer.get_backend("qasm_simulator")
    result = backend.run(transpile(qc, backend), shots=shots).result()
    counts = result.get_counts()

    best_state = max(counts, key=counts.get)
    grover_index = int(best_state, 2)
    if grover_index >= n_candidates:
        grover_index = best_classical

    return grover_index, counts, qc, n_qubits, marked_state, iterations
