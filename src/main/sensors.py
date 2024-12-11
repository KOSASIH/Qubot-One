class Sensor:
    def __init__(self, sensor_type):
        self.sensor_type = sensor_type

    def read_data(self):
        # Simulate reading data from a sensor
        return {"type": self.sensor_type, "value": random.uniform(0, 100)}
