# src/quantum/quantum_networking.py

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import numpy as np
import matplotlib.pyplot as plt

class QuantumNetworking:
    def __init__(self):
        """Initialize the Quantum Networking class with the Aer simulator."""
        self.backend = Aer.get_backend('aer_simulator')

    def create_entanglement_distribution_circuit(self):
        """Create a circuit for entanglement distribution between two parties (Alice and Bob).

        Returns:
            QuantumCircuit: The quantum circuit for entanglement distribution.
        """
        circuit = QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits for measurement
        circuit.h(0)  # Alice prepares a qubit in superposition
        circuit.cx(0, 1)  # Create entanglement with a CNOT gate
        return circuit

    def simulate_entanglement_distribution(self):
        """Simulate the entanglement distribution protocol and visualize the results."""
        circuit = self.create_entanglement_distribution_circuit()
        job = execute(circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector(circuit)

        print("Statevector after entanglement distribution:", statevector)
        self.visualize_state(statevector)

        # Measure the qubits
        circuit.measure([0, 1], [0, 1])
        job = execute(circuit, self.backend, shots=1024)
        counts = job.result().get_counts(circuit)
        print("Measurement Results:", counts)
        self.visualize_results(counts)

    def visualize_state(self, statevector):
        """Visualize the quantum state on the Bloch sphere.

        Args:
            statevector (Statevector): The quantum state to visualize.
        """
        plot_bloch_multivector(statevector)
        plt.title("Quantum State after Entanglement Distribution")
        plt.show()

    def visualize_results(self, counts):
        """Visualize the results of the quantum circuit execution.

        Args:
            counts (dict): The measurement results from the circuit.
        """
        plot_histogram(counts)
        plt.title("Measurement Results from Entanglement Distribution")
        plt.show()

# Example usage
if __name__ == "__main__":
    networking = QuantumNetworking()
    networking.simulate_entanglement_distribution()
