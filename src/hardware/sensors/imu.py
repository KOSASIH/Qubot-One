# src/hardware/sensors/imu.py

import numpy as np
import time

class IMU:
    def __init__(self, imu_id, update_rate=100):
        """Initialize the IMU.

        Args:
            imu_id (str): Identifier for the IMU.
            update_rate (int): Update rate in Hz for data retrieval.
        """
        self.imu_id = imu_id
        self.update_rate = update_rate
        self.orientation = np.zeros(3)  # Roll, Pitch, Yaw
        self.acceleration = np.zeros(3)  # X, Y, Z
        self.gyroscope = np.zeros(3)      # X, Y, Z
        self.temperature = 0.0             # Temperature in Celsius

    def start(self):
        """Start the IMU data acquisition."""
        print(f"Starting IMU {self.imu_id} at {self.update_rate} Hz...")
        # Placeholder for actual initialization logic
        # In a real implementation, this would interface with the IMU hardware

    def stop(self):
        """Stop the IMU data acquisition."""
        print(f"Stopping IMU {self.imu_id}.")

    def read_data(self):
        """Read the latest data from the IMU.

        Returns:
            dict: Dictionary containing orientation, acceleration, gyroscope, and temperature.
        """
        # Placeholder for actual data retrieval logic
        # In a real implementation, this would read from the IMU hardware
        self.orientation = self._simulate_orientation()
        self.acceleration = self._simulate_acceleration()
        self.gyroscope = self._simulate_gyroscope()
        self.temperature = self._simulate_temperature()

        return {
            'orientation': self.orientation,
            'acceleration': self.acceleration,
            'gyroscope': self.gyroscope,
            'temperature': self.temperature
        }

    def _simulate_orientation(self):
        """Simulate orientation data for demonstration purposes.

        Returns:
            np.array: Simulated orientation data (roll, pitch, yaw).
        """
        return np.random.uniform(-180, 180, 3)  # Random orientation in degrees

    def _simulate_acceleration(self):
        """Simulate acceleration data for demonstration purposes.

        Returns:
            np.array: Simulated acceleration data (X, Y, Z).
        """
        return np.random.uniform(-10, 10, 3)  # Random acceleration in m/s^2

    def _simulate_gyroscope(self):
        """Simulate gyroscope data for demonstration purposes.

        Returns:
            np.array: Simulated gyroscope data (X, Y, Z).
        """
        return np.random.uniform(-500, 500, 3)  # Random gyroscope data in degrees/s

    def _simulate_temperature(self):
        """Simulate temperature data for demonstration purposes.

        Returns:
            float: Simulated temperature in degrees Celsius.
        """
        return np.random.uniform(20, 30)  # Random temperature in Celsius

    def get_imu_properties(self):
        """Get the current properties of the IMU.

        Returns:
            dict: Dictionary containing IMU properties.
        """
        properties = {
            'IMU ID': self.imu_id,
            'Update Rate': self.update_rate,
            'Orientation': self.orientation,
            'Acceleration': self.acceleration,
            'Gyroscope': self.gyroscope,
            'Temperature': self.temperature
        }
        return properties

# Example usage
if __name__ == "__main__":
    imu = IMU(imu_id="IMU_1", update_rate=100)
    try:
        imu.start()
        while True:
            data = imu.read_data()
            print("IMU Data:", data)
            time.sleep(1 / imu.update_rate)  # Wait for the next update
    except KeyboardInterrupt:
        imu.stop()
