# src/hardware/communication/can_bus.py

import can

class CANBus:
    def __init__(self, channel, bustype='socketcan'):
        """Initialize the CAN bus communication.

        Args:
            channel (str): CAN channel (e.g., 'can0').
            bustype (str): Type of the CAN bus (default is 'socketcan').
        """
        self.channel = channel
        self.bustype = bustype
        self.bus = can.interface.Bus(channel=self.channel, bustype=self.bustype)

    def send(self, message_id, data):
        """Send a message over the CAN bus.

        Args:
            message_id (int): Identifier for the CAN message.
            data (list): Data to send (list of integers).
        """
        message = can.Message(arbitration_id=message_id, data=data)
        self.bus.send(message)
        print(f"Sent message with ID {message_id}: {data}")

    def receive(self):
        """Receive a message from the CAN bus.

        Returns:
            can.Message: Received CAN message.
        """
        message = self.bus.recv()
        print(f"Received message with ID {message.arbitration_id}: {message.data}")
        return message

    def close(self):
        """Close the CAN bus connection."""
        self.bus.shutdown()
        print(f"CAN bus on {self.channel} closed.")
