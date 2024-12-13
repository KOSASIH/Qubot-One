# src/quantum/quantum_hardware_interface.py

from qiskit import QuantumCircuit, transpile, assemble, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

class QuantumHardwareInterface:
    def __init__(self, provider=None):
        """Initialize the Quantum Hardware Interface.

        Args:
            provider (IBMQProvider): The IBMQ provider for accessing quantum devices.
        """
        if provider is None:
            # Load IBMQ account and get the provider
            IBMQ.load_account()
            self.provider = IBMQ.get_provider(hub='ibm-q')
        else:
            self.provider = provider

    def get_least_busy_device(self):
        """Get the least busy quantum device.

        Returns:
            Backend: The least busy backend available.
        """
        backends = self.provider.backends(filters=lambda x: x.configuration().n_qubits >= 5 and 
                                          not x.configuration().simulator)
        least_busy_backend = least_busy(backends)
        return least_busy_backend

    def run_circuit_on_hardware(self, circuit):
        """Run a quantum circuit on the least busy quantum device.

        Args:
            circuit (QuantumCircuit): The quantum circuit to run.

        Returns:
            dict: The measurement results of the circuit.
        """
        # Get the least busy device
        backend = self.get_least_busy_device()
        print(f"Using backend: {backend.name}")

        # Transpile the circuit for the selected backend
        transpiled_circuit = transpile(circuit, backend)
        qobj = assemble(transpiled_circuit)

        # Execute the circuit on the quantum device
        job = backend.run(qobj)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts

    def visualize_results(self, counts):
        """Visualize the results of the quantum circuit execution.

        Args:
            counts (dict): The measurement results from the circuit.
        """
        plot_histogram(counts)
        plt.title("Measurement Results from Quantum Hardware")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a simple quantum circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)  # Apply Hadamard gate to the first qubit
    circuit.cx(0, 1)  # Apply CNOT gate

    # Initialize the hardware interface
    hardware_interface = QuantumHardwareInterface()

    # Run the circuit on quantum hardware
    counts = hardware_interface.run_circuit_on_hardware(circuit)
    print("Measurement Results:", counts)

    # Visualize the results
    hardware_interface.visualize_results(counts)
