# src/hardware/actuators/gripper.py

class Gripper:
    def __init__(self, gripper_id):
        """Initialize the gripper.

        Args:
            gripper_id (str): Identifier for the gripper.
        """
        self.gripper_id = gripper_id
        self.is_open = True

    def open(self):
        """Open the gripper."""
        self.is_open = True
        print(f"Gripper {self.gripper_id} opened.")

    def close(self):
        """Close the gripper."""
        self.is_open = False
        print(f"Gripper {self.gripper_id} closed.")

    def get_gripper_status(self):
        """Get the current status of the gripper.

        Returns:
            dict: Dictionary containing gripper status.
        """
        return {
            'Gripper ID': self.gripper_id,
            'Is Open': self.is_open
        }
