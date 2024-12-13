# src/middleware/api_gateway.py

from flask import Flask, request, jsonify
from werkzeug.exceptions import NotFound

class APIGateway:
    def __init__(self):
        """Initialize the API gateway."""
        self.app = Flask(__name__)
        self.routes = {}

    def register_route(self, path, service):
        """Register a route with a service.

        Args:
            path (str): The API path.
            service (callable): The service function to handle requests.
        """
        self.routes[path] = service
        self.app.add_url_rule(path, path, self.handle_request, methods=['GET', 'POST'])
        print(f"Route registered: {path}")

    def handle_request(self):
        """Handle incoming requests."""
        path = request.path
        if path not in self.routes:
            raise NotFound(f"Route {path} not found.")
        
        service = self.routes[path]
        if request.method == 'POST':
            data = request.json
            response = service(data)
        else:
            response = service()
        
        return jsonify(response)

    def run(self, host='0.0.0.0', port=5000):
        """Run the API gateway.

        Args:
            host (str): Host to run the API gateway on.
            port (int): Port to run the API gateway on.
        """
        self.app.run(host=host, port=port)
