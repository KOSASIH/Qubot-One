# examples/advanced_usage.py

from src.security import Authenticator
from src.middleware import LoadBalancer

def main():
    # Initialize the authenticator
    authenticator = Authenticator('my_secret_key')

    # Hash and verify a password
    password = 'my_secure_password'
    hashed_password = authenticator.hash_password(password)
    print(f"Hashed Password: {hashed_password}")

    # Initialize the load balancer
    load_balancer = LoadBalancer()
    load_balancer.add_server('http://localhost:5001')
    load_balancer.add_server('http://localhost:5002')

    # Simulate load balancing
    for _ in range(5):
        server = load_balancer.get_server()
        print(f"Routing to server: {server}")

if __name__ == "__main__":
    main()
