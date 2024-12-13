# src/simulation/physics_engine.py

class PhysicsEngine:
    def __init__(self, gravity=(0, -9.81)):
        """Initialize the physics engine.

        Args:
            gravity (tuple): Gravity vector (x, y).
        """
        self.gravity = gravity

    def apply_gravity(self, entity):
        """Apply gravity to an entity.

        Args:
            entity (object): The entity to which gravity will be applied.
        """
        if hasattr(entity, 'velocity'):
            entity.velocity[1] += self.gravity[1]  # Apply gravity to the y-component
            print(f"Gravity applied to {entity}: New velocity: {entity.velocity}")

    def update_physics(self, entities, delta_time):
        """Update the physics for all entities.

        Args:
            entities (list): List of entities to update.
            delta_time (float): Time step for the update.
        """
        for entity in entities:
            if hasattr(entity, 'position') and hasattr(entity, 'velocity'):
                entity.position[0] += entity.velocity[0] * delta_time
                entity.position[1] += entity.velocity[1] * delta_time
                print(f"Updated position for {entity}: New position: {entity.position}")
