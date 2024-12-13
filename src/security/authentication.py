# src/security/authentication.py

import bcrypt
import jwt
from datetime import datetime, timedelta

class Authenticator:
    def __init__(self, secret_key, token_expiration_minutes=30):
        """Initialize the authenticator.

        Args:
            secret_key (str): Secret key for JWT encoding/decoding.
            token_expiration_minutes (int): Token expiration time in minutes.
        """
        self.secret_key = secret_key
        self.token_expiration_minutes = token_expiration_minutes

    def hash_password(self, password):
        """Hash a password.

        Args:
            password (str): The password to hash.

        Returns:
            str: The hashed password.
        """
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        return hashed.decode('utf-8')

    def verify_password(self, password, hashed):
        """Verify a password against a hashed password.

        Args:
            password (str): The password to verify.
            hashed (str): The hashed password.

        Returns:
            bool: True if the password matches, False otherwise.
        """
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def generate_token(self, user_id):
        """Generate a JWT token.

        Args:
            user_id (str): The user ID to include in the token.

        Returns:
            str: The generated JWT token.
        """
        expiration = datetime.utcnow() + timedelta(minutes=self.token_expiration_minutes)
        token = jwt.encode({'user_id': user_id, 'exp': expiration}, self.secret_key, algorithm='HS256')
        return token

    def decode_token(self, token):
        """Decode a JWT token.

        Args:
            token (str): The JWT token to decode.

        Returns:
            dict: The decoded token payload.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
