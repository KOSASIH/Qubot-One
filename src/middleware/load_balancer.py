# src/middleware/load_balancer.py

import random

class LoadBalancer:
    def __init__(self):
        """Initialize the load balancer."""
        self.servers = []

    def add_server(self, server):
        """Add a server to the load balancer.

        Args:
            server (str): The server address (e.g., 'http://localhost:5001').
        """
        self.servers.append(server)
        print(f"Server added: {server}")

    def remove_server(self, server):
        """Remove a server from the load balancer.

        Args:
            server (str): The server address to remove.
        """
        self.servers.remove(server)
        print(f"Server removed: {server}")

    def get_server(self):
        """Get a server using round-robin or random selection.

        Returns:
            str: The selected server address.
        """
        if not self.servers:
            raise Exception("No servers available.")
        selected_server = random.choice(self.servers)
        print(f"Selected server: {selected_server}")
        return selected_server
