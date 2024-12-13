# src/utils/visualization.py

import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_data(x, y, title="Data Plot", xlabel="X-axis", ylabel="Y-axis"):
        """Plot data using Matplotlib.

        Args:
            x (list): X-axis data.
            y (list): Y-axis data.
            title (str): Title of the plot.
            xlabel (str): Label for the X-axis.
            ylabel (str): Label for the Y-axis.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()
        print("Data plotted.")
