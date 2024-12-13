# src/quantum/quantum_fidelity_analysis.py

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, fidelity
import numpy as np
import matplotlib.pyplot as plt

class QuantumFidelityAnalysis:
    def __init__(self):
        """Initialize the Quantum Fidelity Analysis class with the Aer simulator."""
        self.backend = Aer.get_backend('aer_simulator')

    def create_state(self, circuit):
        """Simulate a quantum circuit and return the resulting state vector.

        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate.

        Returns:
            Statevector: The resulting state vector.
        """
        job = execute(circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector(circuit)
        return statevector

    def calculate_fidelity(self, state1, state2):
        """Calculate the fidelity between two quantum states.

        Args:
            state1 (Statevector): The first quantum state.
            state2 (Statevector): The second quantum state.

        Returns:
            float: The fidelity between the two states.
        """
        return fidelity(state1, state2)

    def create_example_states(self):
        """Create example quantum states for fidelity analysis.

        Returns:
            tuple: Two quantum states as Statevectors.
        """
        # Create a circuit for the first state |0> + |1>
        circuit1 = QuantumCircuit(1)
        circuit1.h(0)  # Create superposition state
        state1 = self.create_state(circuit1)

        # Create a circuit for the second state |0> - |1>
        circuit2 = QuantumCircuit(1)
        circuit2.h(0)  # Create superposition state
        circuit2.z(0)  # Apply Z gate to create |0> - |1>
        state2 = self.create_state(circuit2)

        return state1, state2

    def analyze_fidelity(self):
        """Perform fidelity analysis between two quantum states and visualize the results."""
        state1, state2 = self.create_example_states()
        fidelity_value = self.calculate_fidelity(state1, state2)

        print(f"State 1: {state1}")
        print(f"State 2: {state2}")
        print(f"Fidelity between the two states: {fidelity_value:.4f}")

        # Visualize the states on the Bloch sphere
        self.visualize_states(state1, state2)

    def visualize_states(self, state1, state2):
        """Visualize the quantum states on the Bloch sphere.

        Args:
            state1 (Statevector): The first quantum state.
            state2 (Statevector): The second quantum state.
        """
        fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

        # Plot state 1
        ax[0].set_title("State 1")
        self.plot_bloch_vector(state1, ax[0])

        # Plot state 2
        ax[1].set_title("State 2")
        self.plot_bloch_vector(state2, ax[1])

        plt.show()

    def plot_bloch_vector(self, state, ax):
        """Plot a Bloch vector representation of a quantum state.

        Args:
            state (Statevector): The quantum state to plot.
            ax (Axes3D): The axes to plot on.
        """
        # Convert state to Bloch vector
        bloch_vector = state.to_bloch()
        ax.quiver(0, 0, 0, bloch_vector[0], bloch_vector[1], bloch_vector[2], color='b', length=1, normalize=True)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

# Example usage
if __name__ == "__main__":
    fidelity_analysis = QuantumFidelityAnalysis()
    fidelity_analysis.analyze_fidelity()
