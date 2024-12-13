# src/hardware/actuators/servo.py

class Servo:
    def __init__(self, servo_id, min_angle=0, max_angle=180):
        """Initialize the servo motor.

        Args:
            servo_id (str): Identifier for the servo motor.
            min_angle (int): Minimum angle of the servo.
            max_angle (int): Maximum angle of the servo.
        """
        self.servo_id = servo_id
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.current_angle = min_angle

    def set_angle(self, angle):
        """Set the angle of the servo motor.

        Args:
            angle (int): Angle to set (between min_angle and max_angle).
        """
        if self.min_angle <= angle <= self.max_angle:
            self.current_angle = angle
            print(f"Servo {self.servo_id} angle set to {self.current_angle} degrees.")
        else:
            raise ValueError(f"Angle must be between {self.min_angle} and {self.max_angle} degrees.")

    def get_servo_properties(self):
        """Get the current properties of the servo motor.

        Returns:
            dict: Dictionary containing servo properties.
        """
        return {
            'Servo ID': self.servo_id,
            'Min Angle': self.min_angle,
            'Max Angle': self.max_angle,
            'Current Angle': self.current_angle
}
