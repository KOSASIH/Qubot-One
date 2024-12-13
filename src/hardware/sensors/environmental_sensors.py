# src/hardware/sensors/environmental_sensors.py

import random
import time

class EnvironmentalSensors:
    def __init__(self, sensor_id):
        """Initialize the environmental sensor.

        Args:
            sensor_id (str): Identifier for the environmental sensor.
        """
        self.sensor_id = sensor_id
        self.temperature = 0.0  # Temperature in degrees Celsius
        self.humidity = 0.0     # Humidity in percentage
        self.pressure = 0.0     # Pressure in hPa
        self.air_quality = 0.0   # Air quality index (AQI)

    def start(self):
        """Start the environmental sensor data acquisition."""
        print(f"Starting environmental sensor {self.sensor_id}...")

    def stop(self):
        """Stop the environmental sensor data acquisition."""
        print(f"Stopping environmental sensor {self.sensor_id}.")

    def read_data(self):
        """Read the latest data from the environmental sensors.

        Returns:
            dict: Dictionary containing temperature, humidity, pressure, and air quality.
        """
        self.temperature = self._simulate_temperature()
        self.humidity = self._simulate_humidity()
        self.pressure = self._simulate_pressure()
        self.air_quality = self._simulate_air_quality()

        return {
            'temperature': self.temperature,
            'humidity': self.humidity,
            'pressure': self.pressure,
            'air_quality': self.air_quality
        }

    def _simulate_temperature(self):
        """Simulate temperature data for demonstration purposes.

        Returns:
            float: Simulated temperature in degrees Celsius.
        """
        return round(random.uniform(-10, 40), 2)  # Random temperature between -10 and 40 degrees Celsius

    def _simulate_humidity(self):
        """Simulate humidity data for demonstration purposes.

        Returns:
            float: Simulated humidity in percentage.
        """
        return round(random.uniform(0, 100), 2)  # Random humidity between 0% and 100%

    def _simulate_pressure(self):
        """Simulate pressure data for demonstration purposes.

        Returns:
            float: Simulated pressure in hPa.
        """
        return round(random.uniform(950, 1050), 2)  # Random pressure between 950 hPa and 1050 hPa

    def _simulate_air_quality(self):
        """Simulate air quality data for demonstration purposes.

        Returns:
            float: Simulated air quality index (AQI).
        """
        return round(random.uniform(0, 500), 2)  # Random AQI between 0 and 500

    def get_sensor_properties(self):
        """Get the current properties of the environmental sensor.

        Returns:
            dict: Dictionary containing sensor properties.
        """
        properties = {
            'Sensor ID': self.sensor_id,
            'Temperature': self.temperature,
            'Humidity': self.humidity,
            'Pressure': self.pressure,
            'Air Quality': self.air_quality
        }
        return properties

# Example usage
if __name__ == "__main__":
    sensor = EnvironmentalSensors(sensor_id="ENV_SENSOR_1")
    try:
        sensor.start()
        while True:
            data = sensor.read_data()
            print("Environmental Sensor Data:", data)
            time.sleep(1)  # Wait for 1 second before the next reading
    except KeyboardInterrupt:
        sensor.stop()
