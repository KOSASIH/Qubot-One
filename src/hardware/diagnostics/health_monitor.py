# src/hardware/diagnostics/health_monitor.py

class HealthMonitor:
    def __init__(self):
        """Initialize the health monitor."""
        self.health_status = {
            'battery': True,
            'motors': True,
            'sensors': True,
            'communication': True
        }

    def check_battery(self, battery_status):
        """Check the battery health.

        Args:
            battery_status (bool): True if battery is healthy, False otherwise.
        """
        self.health_status['battery'] = battery_status
        print(f"Battery health status: {'Healthy' if battery_status else 'Faulty'}")

    def check_motors(self, motors_status):
        """Check the motors health.

        Args:
            motors_status (bool): True if motors are healthy, False otherwise.
        """
        self.health_status['motors'] = motors_status
        print(f"Motors health status: {'Healthy' if motors_status else 'Faulty'}")

    def check_sensors(self, sensors_status):
        """Check the sensors health.

        Args:
            sensors_status (bool): True if sensors are healthy, False otherwise.
        """
        self.health_status['sensors'] = sensors_status
        print(f"Sensors health status: {'Healthy' if sensors_status else 'Faulty'}")

    def check_communication(self, communication_status):
        """Check the communication health.

        Args:
            communication_status (bool): True if communication is healthy, False otherwise.
        """
        self.health_status['communication'] = communication_status
        print(f"Communication health status: {'Healthy' if communication_status else 'Faulty'}")

    def get_health_status(self):
        """Get the overall health status.

        Returns:
            dict: Dictionary containing health status of all components.
        """
        return self.health_status
