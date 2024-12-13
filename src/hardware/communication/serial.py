# src/hardware/communication/serial.py

import serial

class SerialCommunication:
    def __init__(self, port, baudrate=9600, timeout=1):
        """Initialize the serial communication.

        Args:
            port (str): Serial port (e.g., 'COM3' or '/dev/ttyUSB0').
            baudrate (int ): Baud rate for the serial communication.
            timeout (int): Read timeout in seconds.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = serial.Serial(port, baudrate, timeout=timeout)

    def send(self, data):
        """Send data over the serial connection.

        Args:
            data (str): Data to send.
        """
        self.serial.write(data.encode())
        print(f"Sent: {data}")

    def receive(self):
        """Receive data from the serial connection.

        Returns:
            str: Received data.
        """
        data = self.serial.readline().decode().strip()
        print(f"Received: {data}")
        return data

    def close(self):
        """Close the serial connection."""
        self.serial.close()
        print(f"Serial connection on {self.port} closed.")
