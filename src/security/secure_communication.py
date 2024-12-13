# src/security/secure_communication.py

import socket
import ssl

class SecureCommunication:
    def __init__(self, host, port):
        """Initialize secure communication.

        Args:
            host (str): The host for the secure connection.
            port (int): The port for the secure connection.
        """
        self.host = host
        self.port = port
        self.context = ssl.create_default_context()

    def create_secure_socket(self):
        """Create a secure socket connection.

        Returns:
            socket: The secure socket.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        secure_sock = self.context.wrap_socket(sock, server_hostname=self.host)
        secure_sock.connect((self.host, self.port))
        print(f"Secure socket created and connected to {self.host}:{self.port}")
        return secure_sock

    def send_message(self, message, secure_sock):
        """Send a message over the secure socket.

        Args:
            message (str): The message to send.
            secure_sock (socket): The secure socket to use.
        """
        secure_sock.sendall(message.encode('utf-8'))
        print(f"Message sent: {message}")

    def receive_message(self, secure_sock, buffer_size=1024):
        """Receive a message over the secure socket.

        Args:
            secure_sock (socket): The secure socket to use.
            buffer_size (int): The buffer size for receiving messages.

        Returns:
            str: The received message.
        """
        data = secure_sock.recv(buffer_size)
        message = data.decode('utf-8')
        print(f"Message received: {message}")
        return message
