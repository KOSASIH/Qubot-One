# src/hardware/actuators/motor.py

class Motor:
    def __init__(self, motor_id, max_speed=100):
        """Initialize the motor.

        Args:
            motor_id (str): Identifier for the motor.
            max_speed (int): Maximum speed of the motor.
        """
        self.motor_id = motor_id
        self.max_speed = max_speed
        self.current_speed = 0

    def set_speed(self, speed):
        """Set the speed of the motor.

        Args:
            speed (int): Speed to set (between -max_speed and max_speed).
        """
        if -self.max_speed <= speed <= self.max_speed:
            self.current_speed = speed
            print(f"Motor {self.motor_id} speed set to {self.current_speed}.")
        else:
            raise ValueError(f"Speed must be between {-self.max_speed} and {self.max_speed}.")

    def stop(self):
        """Stop the motor."""
        self.set_speed(0)
        print(f"Motor {self.motor_id} stopped.")

    def get_motor_properties(self):
        """Get the current properties of the motor.

        Returns:
            dict: Dictionary containing motor properties.
        """
        return {
            'Motor ID': self.motor_id,
            'Max Speed': self.max_speed,
            'Current Speed': self.current_speed
        }
