# tests/test_integration.py

import unittest
from src.middleware import APIGateway  # Assuming you have an APIGateway class
from src.security import Authenticator

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.gateway = APIGateway()
        self.authenticator = Authenticator('secret_key')

    def test_api_authentication(self):
        # Simulate an API call with authentication
        token = self.authenticator.generate_token('user_id')
        response = self.gateway.handle_request(token=token)  # Adjust based on your API design
        self.assertEqual(response.status_code, 200)  # Assuming 200 is success

    def test_api_access_control(self):
        # Simulate an API call without authentication
        response = self.gateway.handle_request()  # Adjust based on your API design
        self.assertEqual(response.status_code, 403)  # Assuming 403 is forbidden

if __name__ == '__main__':
    unittest.main()
