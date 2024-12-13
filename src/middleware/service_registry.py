# src/middleware/service_registry.py

class ServiceRegistry:
    def __init__(self):
        """Initialize the service registry."""
        self.services = {}

    def register_service(self, service_name, service_info):
        """Register a service in the registry.

        Args:
            service_name (str): The name of the service.
            service_info (dict): Information about the service (e.g., URL, port).
        """
        self.services[service_name] = service_info
        print(f"Service registered: {service_name} -> {service_info}")

    def deregister_service(self, service_name):
        """Deregister a service from the registry.

        Args:
            service_name (str): The name of the service to deregister.
        """
        if service_name in self.services:
            del self.services[service_name]
            print(f"Service deregistered: {service_name}")

    def get_service(self, service_name):
        """Get service information by name.

        Args:
            service_name (str): The name of the service.

        Returns:
            dict: Service information if found, None otherwise.
        """
        return self.services.get(service_name, None)
