# src/hardware/communication/websocket.py

import websocket

class WebSocketClient:
    def __init__(self, url):
        """Initialize the WebSocket client.

        Args:
            url (str): WebSocket server URL.
        """
        self.url = url
        self.ws = None

    def on_message(self, ws, message):
        """Callback for when a message is received."""
        print(f"Message received: {message}")

    def on_error(self, ws, error):
        """Callback for when an error occurs."""
        print(f"Error: {error}")

    def on_close(self, ws):
        """Callback for when the connection is closed."""
        print("WebSocket connection closed.")

    def on_open(self, ws):
        """Callback for when the connection is opened."""
        print("WebSocket connection opened.")

    def connect(self):
        """Connect to the WebSocket server."""
        self.ws = websocket.WebSocketApp(self.url,
                                          on_message=self.on_message,
                                          on_error=self.on_error,
                                          on_close=self.on_close)
        self.ws.on_open = self.on_open
        self.ws.run_forever()

    def send(self, message):
        """Send a message to the WebSocket server.

        Args:
            message (str): The message to send.
        """
        if self.ws:
            self.ws.send(message)

    def close(self):
        """Close the WebSocket connection."""
        if self.ws:
            self.ws.close()
            print("WebSocket connection closed.")
