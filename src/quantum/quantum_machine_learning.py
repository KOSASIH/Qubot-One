# src/quantum/quantum_machine_learning.py

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class QuantumMachineLearning:
    def __init__(self, num_qubits):
        """Initialize the Quantum Machine Learning model.

        Args:
            num_qubits (int): The number of qubits to use in the quantum circuit.
        """
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('aer_simulator')
        self.feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
        self.model = QSVC(feature_map=self.feature_map, quantum_instance=self.backend)

    def generate_data(self, n_samples=100, n_features=2, n_classes=2):
        """Generate synthetic classification data.

        Args:
            n_samples (int): The number of samples to generate.
            n_features (int): The number of features.
            n_classes (int): The number of classes.

        Returns:
            tuple: Features and labels.
        """
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_features)
        return X, y

    def train(self, X_train, y_train):
        """Train the quantum model.

        Args:
            X_train (np.ndarray): The training features.
            y_train (np.ndarray): The training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the trained model.

        Args:
            X_test (np.ndarray): The test features.

        Returns:
            np.ndarray: The predicted labels.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate the model's performance.

        Args:
            X_test (np.ndarray): The test features.
            y_test (np.ndarray): The true labels.

        Returns:
            float: The accuracy of the model.
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def visualize_data(self, X, y):
        """Visualize the generated data.

        Args:
            X (np.ndarray): The features.
            y (np.ndarray): The labels.
        """
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
        plt.title("Generated Data")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

# Example usage
if __name__ == "__main__":
    qml = QuantumMachineLearning(num_qubits=2)

    # Generate synthetic data
    X, y = qml.generate_data(n_samples=200, n_features=2, n_classes=2)

    # Visualize the data
    qml.visualize_data(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the quantum model
    qml.train(X_train, y_train)

    # Evaluate the model
    accuracy = qml.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
