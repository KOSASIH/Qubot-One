# examples/security_example.py

from src.security import Authenticator

def main():
    # Initialize the authenticator with a secret key
    authenticator = Authenticator('my_secret_key')

    # Step 1: Hash a password
    password = 'my_secure_password'
    hashed_password = authenticator.hash_password(password)
    print(f"Hashed Password: {hashed_password}")

    # Step 2: Verify the password
    is_verified = authenticator.verify_password(password, hashed_password)
    print(f"Password Verified: {is_verified}")

    # Step 3: Generate a JWT token for a user
    user_id = 'user_id_123'
    token = authenticator.generate_token(user_id)
    print(f"Generated Token: {token}")

    # Step 4: Decode the token to retrieve the payload
    payload = authenticator.decode_token(token)
    print(f"Decoded Payload: {payload}")

    # Step 5: Attempt to decode an invalid token
    invalid_token = 'invalid_token_example'
    invalid_payload = authenticator.decode_token(invalid_token)
    print(f"Decoded Invalid Token: {invalid_payload}")  # Should return None

    # Step 6: Test token expiration (optional)
    short_lived_authenticator = Authenticator('my_secret_key', token_expiration_minutes=1)
    short_token = short_lived_authenticator.generate_token(user_id)
    print(f"Short-lived Token: {short_token}")

    # Wait for the token to expire
    import time
    time.sleep(61)  # Sleep for longer than the token expiration time

    expired_payload = short_lived_authenticator.decode_token(short_token)
    print(f"Decoded Expired Token: {expired_payload}")  # Should return None

if __name__ == "__main__":
    main()
