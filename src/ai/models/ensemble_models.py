# src/ai/models/ensemble_models.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import joblib

class BaggingModel:
    def __init__(self, n_estimators=100):
        """Initialize the Bagging model using Random Forest.

        Args:
            n_estimators (int): Number of trees in the forest.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, X, y):
        """Train the Bagging model.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Target vector.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the Bagging model.

        Args:
            X (np.array): Feature matrix for predictions.

        Returns:
            np.array: Predicted labels.
        """
        return self.model.predict(X)

    def save_model(self, filepath):
        """Save the trained model to a file.

        Args:
            filepath (str): Path to save the model.
        """
        joblib.dump(self.model, filepath)
        print(f"Bagging model saved to {filepath}")

    def load_model(self, filepath):
        """Load a model from a file.

        Args:
            filepath (str): Path to load the model from.
        """
        self.model = joblib.load(filepath)
        print(f"Bagging model loaded from {filepath}")

class BoostingModel:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        """Initialize the Boosting model using Gradient Boosting.

        Args:
            n_estimators (int): Number of boosting stages to be run.
            learning_rate (float): Learning rate shrinks the contribution of each tree.
        """
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

    def train(self, X, y):
        """Train the Boosting model.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Target vector.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the Boosting model.

        Args:
            X (np.array): Feature matrix for predictions.

        Returns:
            np.array: Predicted labels.
        """
        return self.model.predict(X)

    def save_model(self, filepath):
        """Save the trained model to a file.

        Args:
            filepath (str): Path to save the model.
        """
        joblib.dump(self.model, filepath)
        print(f"Boosting model saved to {filepath}")

    def load_model(self, filepath):
        """Load a model from a file.

        Args:
            filepath (str): Path to load the model from.
        """
        self.model = joblib.load(filepath)
        print(f"Boosting model loaded from {filepath}")

class StackingModel:
    def __init__(self, base_models, final_model):
        """Initialize the Stacking model.

        Args:
            base_models (list): List of base models to be used in stacking.
            final_model (object): Final model to combine the predictions of base models.
        """
        self.model = VotingClassifier(estimators=base_models, voting='soft')
        self.final_model = final_model

    def train(self, X, y):
        """Train the Stacking model.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Target vector.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the Stacking model.

        Args:
            X (np.array): Feature matrix for predictions.

        Returns:
            np.array: Predicted labels.
        """
        return self.model.predict(X)

    def save_model(self, filepath):
        """Save the trained model to a file.

        Args:
            filepath (str): Path to save the model.
        """
        joblib.dump(self.model, filepath)
        print(f"Stacking model saved to {filepath}")

    def load_model(self, filepath):
        """Load a model from a file.

        Args:
            filepath (str): Path to load the model from.
        """
        self.model = joblib.load(filepath)
        print(f"Stacking model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Load dataset
    iris = load
