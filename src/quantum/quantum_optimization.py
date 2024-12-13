# src/quantum/quantum_optimization.py

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.algorithms.optimizers import SPSA
import numpy as np

class QuantumOptimization:
    def __init__(self):
        """Initialize the Quantum Optimization class with the Aer simulator."""
        self.backend = Aer.get_backend('aer_simulator')

    def create_qaoa_circuit(self, p, num_qubits):
        """Create a QAOA circuit.

        Args:
            p (int): The number of layers in the QAOA circuit.
            num_qubits (int): The number of qubits.

        Returns:
            QuantumCircuit: The QAOA circuit.
        """
        circuit = QuantumCircuit(num_qubits)
        # Initialize qubits to |+>
        circuit.h(range(num_qubits))

        # Apply QAOA layers
        for layer in range(p):
            # Apply the cost Hamiltonian (example: simple Ising model)
            for i in range(num_qubits):
                circuit.rz(np.pi / 2, i)  # Example of a rotation
            # Apply the mixing Hamiltonian
            circuit.rx(np.pi / 2, range(num_qubits))

        return circuit

    def run_qaoa(self, p, num_qubits):
        """Run the QAOA algorithm.

        Args:
            p (int): The number of layers in the QAOA circuit.
            num_qubits (int): The number of qubits.

        Returns:
            dict: The measurement results of the QAOA circuit.
        """
        circuit = self.create_qaoa_circuit(p, num_qubits)
        job = execute(circuit, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts

    def run_vqe(self, ansatz, Hamiltonian):
        """Run the Variational Quantum Eigensolver (VQE).

        Args:
            ansatz (QuantumCircuit): The ansatz circuit for VQE.
            Hamiltonian (QuantumCircuit): The Hamiltonian to minimize.

        Returns:
            float: The minimum eigenvalue found by VQE.
        """
        optimizer = SPSA(maxiter=100)
        vqe = VQE(ansatz, optimizer=optimizer)
        result = vqe.compute_minimum_eigenvalue(Hamiltonian)
        return result.eigenvalue

    def create_hamiltonian(self, num_qubits):
        """Create a simple Hamiltonian for testing.

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            QuantumCircuit: The Hamiltonian circuit.
        """
        # Example: Create a Hamiltonian for a simple Ising model
        hamiltonian = QuantumCircuit(num_qubits)
        for i in range(num_qubits - 1):
            hamiltonian.cx(i, i + 1)  # Example of coupling
        return hamiltonian

# Example usage
if __name__ == "__main__":
    optimizer = QuantumOptimization()

    # Run QAOA
    p = 2  # Number of layers
    num_qubits = 3  # Number of qubits
    qaoa_results = optimizer.run_qaoa(p, num_qubits)
    print("QAOA Measurement Results:", qaoa_results)

    # Run VQE
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2)
    hamiltonian = optimizer.create_hamiltonian(num_qubits)
    vqe_result = optimizer.run_vqe(ansatz, hamiltonian)
    print("VQE Minimum Eigenvalue:", vqe_result)
