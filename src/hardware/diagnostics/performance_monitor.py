# src/hardware/diagnostics/performance_monitor.py

import time

class PerformanceMonitor:
    def __init__(self):
        """Initialize the performance monitor."""
        self.start_time = time.time()
        self.data_points = []

    def log_performance(self, data):
        """Log performance data.

        Args:
            data (float): Performance metric (e.g., speed, efficiency).
        """
        self.data_points.append(data)
        print(f"Performance data logged: {data}")

    def calculate_average_performance(self):
        """Calculate the average performance from logged data.

        Returns:
            float: Average performance.
        """
        if not self.data_points:
            return 0return sum(self.data_points) / len(self.data_points)

    def get_performance_report(self):
        """Get a report of the performance metrics.

        Returns:
            dict: Dictionary containing performance metrics.
        """
        average_performance = self.calculate_average_performance()
        elapsed_time = time.time() - self.start_time
        return {
            'Average Performance': average_performance,
            'Total Time (s)': elapsed_time,
            'Data Points': self.data_points
        }
