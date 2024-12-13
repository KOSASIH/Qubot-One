# src/hardware/power_management/energy_harvesting.py

class EnergyHarvesting:
    def __init__(self, source_type):
        """Initialize the energy harvesting system.

        Args:
            source_type (str): Type of energy source (e.g., 'solar', 'wind', 'thermal').
        """
        self.source_type = source_type
        self.energy_collected = 0  # Initial energy collected

    def harvest_energy(self, amount):
        """Harvest energy from the source.

        Args:
            amount (float): Amount of energy harvested in joules (J).
        """
        self.energy_collected += amount
        print(f"Harvested {amount} J from {self.source_type}. Total energy collected: {self.energy_collected} J.")

    def get_energy_status(self):
        """Get the current status of the energy harvesting system.

        Returns:
            dict: Dictionary containing energy harvesting status.
        """
        return {
            'Source Type': self.source_type,
            'Energy Collected (J)': self.energy_collected
        }
