# examples/basic_usage.py

from src.security import Authenticator

def main():
    # Initialize the authenticator
    authenticator = Authenticator('my_secret_key')

    # Hash a password
    password = 'my_password'
    hashed_password = authenticator.hash_password(password)
    print(f"Hashed Password: {hashed_password}")

    # Verify the password
    is_verified = authenticator.verify_password(password, hashed_password)
    print(f"Password Verified: {is_verified}")

    # Generate a token
    token = authenticator.generate_token('user_id_123')
    print(f"Generated Token: {token}")

    # Decode the token
    payload = authenticator.decode_token(token)
    print(f"Decoded Payload: {payload}")

if __name__ == "__main__":
    main()
