# src/simulation/scenario_generator.py

import random

class ScenarioGenerator:
    def __init__(self):
        """Initialize the scenario generator."""
        pass

    def generate_random_scenario(self, num_entities, environment):
        """Generate a random scenario with entities.

        Args:
            num_entities (int): Number of entities to generate.
            environment (SimulationEnvironment): The environment to populate.
        """
        for _ in range(num_entities):
            position = [random.uniform(0, environment.width), random.uniform(0, environment.height)]
            velocity = [random.uniform(-1, 1), random.uniform(-1, 1)]
            entity = {
                'position': position,
                'velocity': velocity
            }
            environment.add_entity(entity)
            print(f"Generated entity: {entity}")
