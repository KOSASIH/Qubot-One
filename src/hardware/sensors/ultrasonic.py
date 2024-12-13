# src/hardware/sensors/ultrasonic.py

import time
import random

class Ultrasonic:
    def __init__(self, sensor_id, max_distance=400, min_distance=2):
        """Initialize the ultrasonic sensor.

        Args:
            sensor_id (str): Identifier for the ultrasonic sensor.
            max_distance (int): Maximum measurable distance in centimeters.
            min_distance (int): Minimum measurable distance in centimeters.
        """
        self.sensor_id = sensor_id
        self.max_distance = max_distance
        self.min_distance = min_distance

    def start(self):
        """Start the ultrasonic sensor."""
        print(f"Starting ultrasonic sensor {self.sensor_id}...")

    def stop(self):
        """Stop the ultrasonic sensor."""
        print(f"Stopping ultrasonic sensor {self.sensor_id}.")

    def get_distance(self):
        """Get the distance measurement from the ultrasonic sensor.

        Returns:
            float: Distance in centimeters.
        """
        # Simulate distance measurement
        distance = self._simulate_distance()
        return distance

    def _simulate_distance(self):
        """Simulate distance measurement for demonstration purposes.

        Returns:
            float: Simulated distance in centimeters.
        """
        # Generate a random distance within the specified range
        return random.uniform(self.min_distance, self.max_distance)

    def get_sensor_properties(self):
        """Get the current properties of the ultrasonic sensor.

        Returns:
            dict: Dictionary containing sensor properties.
        """
        properties = {
            'Sensor ID': self.sensor_id,
            'Max Distance': self.max_distance,
            'Min Distance': self.min_distance
        }
        return properties

# Example usage
if __name__ == "__main__":
    ultrasonic_sensor = Ultrasonic(sensor_id="ULTRASONIC_1", max_distance=400, min_distance=2)
    try:
        ultrasonic_sensor.start()
        while True:
            distance = ultrasonic_sensor.get_distance()
            print(f"Distance measured by {ultrasonic_sensor.sensor_id}: {distance:.2f} cm")
            time.sleep(1)  # Wait for 1 second before the next measurement
    except KeyboardInterrupt:
        ultrasonic_sensor.stop()
