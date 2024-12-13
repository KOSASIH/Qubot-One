# src/quantum/quantum_state_preparation.py

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import numpy as np

class QuantumStatePreparation:
    def __init__(self):
        """Initialize the Quantum State Preparation class with the Aer simulator."""
        self.backend = Aer.get_backend('aer_simulator')

    def prepare_computational_basis_state(self, state_index, num_qubits):
        """Prepare a computational basis state.

        Args:
            state_index (int): The index of the computational basis state to prepare.
            num_qubits (int): The number of qubits.

        Returns:
            QuantumCircuit: The quantum circuit preparing the state.
        """
        circuit = QuantumCircuit(num_qubits)
        # Prepare the state |state_index>
        binary_state = format(state_index, f'0{num_qubits}b')
        for i, bit in enumerate(binary_state):
            if bit == '1':
                circuit.x(i)  # Apply X gate to flip the qubit to |1>
        return circuit

    def prepare_superposition_state(self, num_qubits):
        """Prepare a superposition state (equal superposition of all basis states).

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            QuantumCircuit: The quantum circuit preparing the superposition state.
        """
        circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            circuit.h(i)  # Apply Hadamard gate to each qubit
        return circuit

    def prepare_entangled_state(self, num_qubits):
        """Prepare a maximally entangled state (e.g., GHZ state).

        Args:
            num_qubits (int): The number of qubits.

        Returns:
            QuantumCircuit: The quantum circuit preparing the entangled state.
        """
        circuit = QuantumCircuit(num_qubits)
        circuit.h(0)  # Apply Hadamard to the first qubit
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)  # Apply CNOT gates to create entanglement
        return circuit

    def simulate_and_visualize(self, circuit):
        """Simulate the quantum circuit and visualize the results.

        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate.
        """
        # Simulate the circuit
        job = execute(circuit, self.backend, shots=1024)
        result = job.result()
        statevector = result.get_statevector(circuit)
        
        # Visualize the state on the Bloch sphere
        plot_bloch_multivector(statevector)
        plt.title("Quantum State on Bloch Sphere")
        plt.show()

        # Get measurement results
        counts = result.get_counts(circuit)
        self.visualize_results(counts)

    def visualize_results(self, counts):
        """Visualize the results of the quantum circuit execution.

        Args:
            counts (dict): The measurement results from the circuit.
        """
        plot_histogram(counts)
        plt.title("Measurement Results")
        plt.show()

# Example usage
if __name__ == "__main__":
    state_preparation = QuantumStatePreparation()

    # Prepare and visualize a computational basis state |3> (for 3 qubits)
    circuit1 = state_preparation.prepare_computational_basis_state(3, 3)
    state_preparation.simulate_and_visualize(circuit1)

    # Prepare and visualize a superposition state (3 qubits)
    circuit2 = state_preparation.prepare_superposition_state(3)
    state_preparation.simulate_and_visualize(circuit2)

    # Prepare and visualize a GHZ entangled state (3 qubits)
    circuit3 = state_preparation.prepare_entangled_state(3)
    state_preparation.simulate_and_visualize(circuit3)
