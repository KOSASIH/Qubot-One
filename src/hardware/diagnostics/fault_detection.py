# src/hardware/diagnostics/fault_detection.py

class FaultDetection:
    def __init__(self):
        """Initialize the fault detection system."""
        self.faults = []

    def detect_motor_fault(self, motor_data):
        """Detect faults in the motor based on data.

        Args:
            motor_data (dict): Dictionary containing motor parameters (e.g., speed, temperature).
        """
        if motor_data['temperature'] > 100:  # Example threshold
            self.faults.append('Motor overheating')
            print("Fault detected: Motor overheating.")

        if motor_data['speed'] < 0:  # Example condition
            self.faults.append('Motor speed negative')
            print("Fault detected: Motor speed negative.")

    def detect_sensor_fault(self, sensor_data):
        """Detect faults in the sensors based on data.

        Args:
            sensor_data (dict): Dictionary containing sensor parameters (e.g., readings).
        """
        if sensor_data['reading'] < 0:  # Example condition
            self.faults.append('Sensor reading invalid')
            print("Fault detected: Sensor reading invalid.")

    def get_faults(self):
        """Get the list of detected faults.

        Returns:
            list: List of detected faults.
        """
        return self.faults
