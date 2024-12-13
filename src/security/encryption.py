# src/security/encryption.py

from cryptography.fernet import Fernet

class EncryptionManager:
    def __init__(self, key=None):
        """Initialize the encryption manager.

        Args:
            key (bytes): The encryption key. If None, a new key will be generated.
        """
        if key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = key
        self.cipher = Fernet(self.key)

    def encrypt(self, data):
        """Encrypt data.

        Args:
            data (str): The data to encrypt.

        Returns:
            bytes: The encrypted data.
        """
        return self.cipher.encrypt(data.encode('utf-8'))

    def decrypt(self, encrypted_data):
        """Decrypt data.

        Args:
            encrypted_data (bytes): The data to decrypt.

        Returns:
            str: The decrypted data.
        """
        return self.cipher.decrypt(encrypted_data).decode('utf-8')

    def get_key(self):
        """Get the encryption key.

        Returns:
            bytes: The encryption key.
        """
        return self.key
