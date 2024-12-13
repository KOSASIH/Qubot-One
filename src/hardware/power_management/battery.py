# src/hardware/power_management/battery.py

class BatteryManagement:
    def __init__(self, capacity, voltage):
        """Initialize the battery management system.

        Args:
            capacity (float): Battery capacity in ampere-hours (Ah).
            voltage (float): Battery voltage in volts (V).
        """
        self.capacity = capacity  # in Ah
        self.voltage = voltage      # in V
        self.current_charge = capacity  # Start fully charged

    def discharge(self, current, time_hours):
        """Discharge the battery.

        Args:
            current (float): Current in amperes (A).
            time_hours (float): Time in hours for which the current is drawn.
        """
        charge_used = current * time_hours
        if charge_used > self.current_charge:
            raise ValueError("Not enough charge to discharge.")
        self.current_charge -= charge_used
        print(f"Discharged {charge_used:.2f} Ah. Current charge: {self.current_charge:.2f} Ah.")

    def charge(self, current, time_hours):
        """Charge the battery.

        Args:
            current (float): Charging current in amperes (A).
            time_hours (float): Time in hours for which the battery is charged.
        """
        charge_added = current * time_hours
        if self.current_charge + charge_added > self.capacity:
            raise ValueError("Charging exceeds battery capacity.")
        self.current_charge += charge_added
        print(f"Charged {charge_added:.2f} Ah. Current charge: {self.current_charge:.2f} Ah.")

    def get_battery_status(self):
        """Get the current status of the battery.

        Returns:
            dict: Dictionary containing battery status.
        """
        return {
            'Capacity (Ah)': self.capacity,
            'Voltage (V)': self.voltage,
            'Current Charge (Ah)': self.current_charge
        }
