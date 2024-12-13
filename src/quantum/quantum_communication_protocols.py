# src/quantum/quantum_communication_protocols.py

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import numpy as np
import matplotlib.pyplot as plt

class QuantumCommunicationProtocols:
    def __init__(self):
        """Initialize the Quantum Communication Protocols class with the Aer simulator."""
        self.backend = Aer.get_backend('aer_simulator')

    def create_bb84_protocol(self):
        """Create a BB84 quantum key distribution protocol circuit.

        Returns:
            QuantumCircuit: The quantum circuit implementing the BB84 protocol.
        """
        # Alice's preparation of qubits
        circuit = QuantumCircuit(2, 1)  # 2 qubits, 1 classical bit for measurement
        # Alice randomly chooses a basis and prepares a qubit
        basis_choice = np.random.choice(['X', 'Z'])  # Randomly choose basis
        if basis_choice == 'X':
            circuit.h(0)  # Prepare in the X basis
        else:
            circuit.z(0)  # Prepare in the Z basis (if needed)

        # Measure the qubit
        circuit.measure(0, 0)  # Measure the first qubit

        return circuit, basis_choice

    def simulate_bb84(self):
        """Simulate the BB84 protocol and visualize the results."""
        circuit, basis_choice = self.create_bb84_protocol()
        job = execute(circuit, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)
        print(f"BB84 Protocol - Basis Choice: {basis_choice}")
        print("Measurement Results:", counts)
        self.visualize_results(counts)

    def visualize_results(self, counts):
        """Visualize the results of the simulation.

        Args:
            counts (dict): The measurement results from the simulation.
        """
        plot_histogram(counts)
        plt.title("BB84 Measurement Results")
        plt.show()

    def create_quantum_teleportation_circuit(self):
        """Create a quantum teleportation circuit.

        Returns:
            QuantumCircuit: The quantum circuit implementing quantum teleportation.
        """
        circuit = QuantumCircuit(3, 3)  # 3 qubits, 3 classical bits for measurement
        # Create entanglement between qubit 1 and qubit 2
        circuit.h(1)  # Create a Bell pair
        circuit.cx(1, 2)

        # Alice's qubit (0) to be teleported
        circuit.cx(0, 1)  # CNOT
        circuit.h(0)      # Hadamard

        # Measure Alice's qubits
        circuit.measure(0, 0)  # Measure qubit 0
        circuit.measure(1, 1)  # Measure qubit 1

        # Apply corrections based on measurements
        circuit.cx(1, 2)  # If the first measurement is 1, apply CNOT
        circuit.cz(0, 2)  # If the second measurement is 1, apply CZ

        return circuit

    def simulate_quantum_teleportation(self):
        """Simulate the quantum teleportation protocol and visualize the results."""
        circuit = self.create_quantum_teleportation_circuit()
        job = execute(circuit, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)
        print("Quantum Teleportation Measurement Results:", counts)
        self.visualize_results(counts)

# Example usage
if __name__ == "__main__":
    protocols = QuantumCommunicationProtocols()
    protocols.simulate_bb84()
    protocols.simulate_quantum_teleportation()
