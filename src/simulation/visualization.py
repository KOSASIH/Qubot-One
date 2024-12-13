# src/simulation/visualization.py

class Visualization:
    def __init__(self, environment):
        """Initialize the visualization tools.

        Args:
            environment (SimulationEnvironment): The environment to visualize.
        """
        self.environment = environment

    def render(self):
        """Render the current state of the environment."""
        print("Rendering environment:")
        for entity in self.environment.get_entities():
            print(f"Entity at position: {entity['position']} with velocity: {entity['velocity']}")

    def update_display(self):
        """Update the display with the latest simulation state."""
        print("Updating display...")
        self.render()
