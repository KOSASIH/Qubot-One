# examples/integration_example.py

from src.middleware import APIGateway
from src.security import Authenticator

def main():
    # Initialize the API gateway and authenticator
    gateway = APIGateway()
    authenticator = Authenticator('my_secret_key')

    # Simulate user authentication
    token = authenticator.generate_token('user_id_123')
    print(f"User Token: {token}")

    # Simulate an API request with the token
    response = gateway.handle_request(token=token)
    print(f"API Response: {response}")

if __name__ == "__main__":
    main()
