# src/hardware/power_management/power_distribution.py

class PowerDistribution:
    def __init__(self):
        """Initialize the power distribution system."""
        self.devices = {}

    def add_device(self, device_id, power_rating):
        """Add a device to the power distribution system.

        Args:
            device_id (str): Identifier for the device.
            power_rating (float): Power rating of the device in watts (W).
        """
        self.devices[device_id] = {
            'power_rating': power_rating,
            'status': 'off'
        }
        print(f"Device {device_id} added with power rating {power_rating} W.")

    def turn_on_device(self, device_id):
        """Turn on a device.

        Args:
            device_id (str): Identifier for the device.
        """
        if device_id in self.devices:
            self.devices[device_id]['status'] = 'on'
            print(f"Device {device_id} turned on.")
        else:
            raise ValueError("Device not found.")

    def turn_off_device(self, device_id):
        """Turn off a device.

        Args:
            device_id (str): Identifier for the device.
        """
        if device_id in self.devices:
            self.devices[device_id]['status'] = 'off'
            print(f"Device {device_id} turned off.")
        else:
            raise ValueError("Device not found.")

    def get_power_distribution_status(self):
        """Get the current status of all devices in the power distribution system.

        Returns:
            dict: Dictionary containing the status of all devices.
        """
        return {device_id: info for device_id, info in self.devices.items()}
