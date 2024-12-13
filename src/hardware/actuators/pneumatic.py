# src/hardware/actuators/pneumatic.py

class Pneumatic:
    def __init__(self, actuator_id):
        """Initialize the pneumatic actuator.

        Args:
            actuator_id (str): Identifier for the pneumatic actuator.
        """
        self.actuator_id = actuator_id
        self.is_active = False

    def activate(self):
        """Activate the pneumatic actuator."""
        self.is_active = True
        print(f"Pneumatic actuator {self.actuator_id} activated.")

    def deactivate(self):
        """Deactivate the pneumatic actuator."""
        self.is_active = False
        print(f"Pneumatic actuator {self.actuator_id} deactivated.")

    def get_actuator_status(self):
        """Get the current status of the pneumatic actuator.

        Returns:
            dict: Dictionary containing actuator status.
        """
        return {
            'Actuator ID': self.actuator_id,
            'Is Active': self.is_active
        }
