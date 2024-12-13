# src/quantum/quantum_simulation.py

from qiskit import Aer, QuantumCircuit, execute
from qiskit.visualization import plot_histogram, plot_state_qsphere
import numpy as np
import matplotlib.pyplot as plt

class QuantumSimulation:
    def __init__(self):
        """Initialize the Quantum Simulation class with the Aer simulator."""
        self.backend = Aer.get_backend('aer_simulator')

    def simulate_circuit(self, circuit):
        """Simulate a quantum circuit and return the results.

        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate.

        Returns:
            dict: The measurement results of the circuit.
        """
        # Execute the circuit on the Aer simulator
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

    def simulate_statevector(self, circuit):
        """Simulate the state vector of a quantum circuit.

        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate.

        Returns:
            np.ndarray: The state vector resulting from the circuit execution.
        """
        # Execute the circuit on the statevector simulator
        job = execute(circuit, self.backend, shots=1, memory=True)
        result = job.result()
        statevector = result.get_statevector(circuit)
        return statevector

    def visualize_state(self, statevector):
        """Visualize the quantum state using a Q-sphere representation.

        Args:
            statevector (np.ndarray): The state vector to visualize.
        """
        plot_state_qsphere(statevector)
        plt.title("Quantum State Q-sphere")
        plt.show()

    def run_example_circuit(self):
        """Run an example quantum circuit and visualize the results."""
        # Create a simple quantum circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)  # Apply Hadamard gate to the first qubit
        circuit.cx(0, 1)  # Apply CNOT gate

        # Simulate the circuit
        counts = self.simulate_circuit(circuit)
        print("Measurement Results:", counts)
        self.visualize_results(counts)

        # Simulate the state vector
        statevector = self.simulate_statevector(circuit)
        print("State Vector:", statevector)
        self.visualize_state(statevector)

# Example usage
if __name__ == "__main__":
    simulation = QuantumSimulation()
    simulation.run_example_circuit()
