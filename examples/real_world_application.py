# examples/real_world_application.py

from src.security import Authenticator
from src.middleware import APIGateway

def main():
    # Initialize components
    authenticator = Authenticator('my_secret_key')
    gateway = APIGateway()

    # User login simulation
    username = 'user1'
    password = 'password123'
    hashed_password = authenticator.hash_password(password)

    # Simulate user authentication
    if authenticator.verify_password(password, hashed_password):
        token = authenticator.generate_token(username)
        print(f"User {username} authenticated. Token: {token}")

        # Simulate API request
        response = gateway.handle_request(token=token)
        print(f"API Response: {response}")
    else:
        print("Authentication failed.")

if __name__ == "__main__":
    main()
