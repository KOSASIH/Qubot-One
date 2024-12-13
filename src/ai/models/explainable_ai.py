# src/ai/models/explainable_ai.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ExplainableAI:
    def __init__(self, model):
        """Initialize the Explainable AI class.

        Args:
            model (object): The trained model to explain.
        """
        self.model = model

    def explain_with_shap(self, X, feature_names):
        """Generate SHAP explanations for the model predictions.

        Args:
            X (np.array): Feature matrix for which to explain predictions.
            feature_names (list): List of feature names.

        Returns:
            shap.Explanation: SHAP explanation object.
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values

    def explain_with_lime(self, X, feature_names, instance):
        """Generate LIME explanations for a single instance.

        Args:
            X (np.array): Feature matrix for which to explain predictions.
            feature_names (list): List of feature names.
            instance (np.array): The instance to explain.

        Returns:
            lime.lime_tabular.LimeTabularExplainer: LIME explainer object.
        """
        explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=feature_names, class_names=['0', '1', '2'], mode='classification')
        exp = explainer.explain_instance(instance, self.model.predict_proba, num_features=len(feature_names))
        return exp

    def plot_shap_values(self, shap_values, feature_names):
        """Plot SHAP values.

        Args:
            shap_values (np.array): SHAP values to plot.
            feature_names (list): List of feature names.
        """
        shap.summary_plot(shap_values, feature_names=feature_names)

    def plot_lime_explanation(self, exp):
        """Plot LIME explanation.

        Args:
            exp (lime.lime_tabular.LimeTabularExplainer): LIME explanation object.
        """
        exp.as_pyplot_figure()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Initialize Explainable AI
    explainable_ai = ExplainableAI(model)

    # Generate SHAP explanations
    shap_values = explainable_ai.explain_with_shap(X_test, feature_names)
    explainable_ai.plot_shap_values(shap_values, feature_names)

    # Generate LIME explanation for a single instance
    instance = X_test[0]
    lime_exp = explainable_ai.explain_with_lime(X_train, feature_names, instance)
    explainable_ai.plot_lime_explanation(lime_exp)
