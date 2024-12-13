# tests/test_security.py

import unittest
from src.security import Authenticator  # Assuming you have an Authenticator class

class TestSecurity(unittest.TestCase):
    def setUp(self):
        """Set up the test environment for each test case."""
        self.authenticator = Authenticator('secret_key')

    def test_password_hashing(self):
        """Test that password hashing works correctly."""
        password = 'my_password'
        hashed = self.authenticator.hash_password(password)
        self.assertNotEqual(password, hashed)  # Ensure the hashed password is not the same as the original
        self.assertTrue(self.authenticator.verify_password(password, hashed))  # Verify the password matches the hash

    def test_token_generation(self):
        """Test that a token can be generated successfully."""
        token = self.authenticator.generate_token('user_id')
        self.assertIsNotNone(token)  # Ensure the token is not None
        self.assertIsInstance(token, str)  # Ensure the token is a string

    def test_token_decoding(self):
        """Test that a token can be decoded successfully."""
        token = self.authenticator.generate_token('user_id')
        payload = self.authenticator.decode_token(token)
        self.assertEqual(payload['user_id'], 'user_id')  # Ensure the user_id in the payload matches

    def test_invalid_token(self):
        """Test that decoding an invalid token returns None."""
        payload = self.authenticator.decode_token('invalid_token')
        self.assertIsNone(payload)  # Ensure that decoding an invalid token returns None

    def test_expired_token(self):
        """Test that an expired token cannot be decoded."""
        # Generate a token with a short expiration time for testing
        short_lived_authenticator = Authenticator('secret_key', token_expiration_minutes=1)
        token = short_lived_authenticator.generate_token('user_id')
        
        # Wait for the token to expire
        import time
        time.sleep(61)  # Sleep for longer than the token expiration time

        payload = short_lived_authenticator.decode_token(token)
        self.assertIsNone(payload)  # Ensure that the expired token returns None

if __name__ == '__main__':
    unittest.main()
