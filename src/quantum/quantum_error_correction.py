# src/quantum/quantum_error_correction.py

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

class QuantumErrorCorrection:
    def __init__(self):
        """Initialize the Quantum Error Correction class."""
        self.backend = Aer.get_backend('aer_simulator')

    def create_shor_code(self):
        """Create a Shor error correction code circuit.

        Returns:
            QuantumCircuit: The quantum circuit implementing the Shor code.
        """
        circuit = QuantumCircuit(9, 3)
        # Encoding
        circuit.h(0)  # Prepare the state |+>
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(1, 3)
        circuit.cx(1, 4)
        circuit.cx(2, 5)
        circuit.cx(2, 6)
        circuit.cx(3, 7)
        circuit.cx(3, 8)

        # Measurement
        circuit.measure([0, 1, 2], [0, 1, 2])  # Measure the first three qubits
        return circuit

    def create_repetition_code(self, qubit):
        """Create a repetition code circuit for a single qubit.

        Args:
            qubit (int): The qubit to encode.

        Returns:
            QuantumCircuit: The quantum circuit implementing the repetition code.
        """
        circuit = QuantumCircuit(3, 1)
        circuit.cx(qubit, 0)
        circuit.cx(qubit, 1)
        circuit.measure(0, 0)
        circuit.measure(1, 0)
        return circuit

    def simulate_circuit(self, circuit):
        """Simulate a quantum circuit and return the results.

        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate.

        Returns:
            dict: The measurement results of the circuit.
        """
        job = execute(circuit, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts

    def visualize_results(self, counts):
        """Visualize the results of the simulation.

        Args:
            counts (dict): The measurement results from the simulation.
        """
        plot_histogram(counts)
        plt.title("Measurement Results")
        plt.show()

    def run_shor_code(self):
        """Run the Shor error correction code and visualize the results."""
        circuit = self.create_shor_code()
        counts = self.simulate_circuit(circuit)
        print("Shor Code Measurement Results:", counts)
        self.visualize_results(counts)

    def run_repetition_code(self):
        """Run the repetition code and visualize the results."""
        circuit = self.create_repetition_code(0)  # Encode the first qubit
        counts = self.simulate_circuit(circuit)
        print("Repetition Code Measurement Results:", counts)
        self.visualize_results(counts)

# Example usage
if __name__ == "__main__":
    error_correction = QuantumErrorCorrection()
    error_correction.run_shor_code()
    error_correction.run_repetition_code()
