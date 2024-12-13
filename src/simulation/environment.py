# src/simulation/environment.py

class SimulationEnvironment:
    def __init__(self, width, height):
        """Initialize the simulation environment.

        Args:
            width (int): Width of the environment.
            height (int): Height of the environment.
        """
        self.width = width
        self.height = height
        self.entities = []  # List to hold entities in the environment

    def add_entity(self, entity):
        """Add an entity to the environment.

        Args:
            entity (object): The entity to add.
        """
        self.entities.append(entity)
        print(f"Entity {entity} added to the environment.")

    def remove_entity(self, entity):
        """Remove an entity from the environment.

        Args:
            entity (object): The entity to remove.
        """
        self.entities.remove(entity)
        print(f"Entity {entity} removed from the environment.")

    def get_entities(self):
        """Get the list of entities in the environment.

        Returns:
            list: List of entities.
        """
        return self.entities
