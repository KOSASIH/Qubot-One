# src/hardware/communication/mqtt.py

import paho.mqtt.client as mqtt

class MQTTClient:
    def __init__(self, broker, port=1883, client_id=""):
        """Initialize the MQTT client.

        Args:
            broker (str): MQTT broker address.
            port (int): Port number for the MQTT broker.
            client_id (str): Client ID for the MQTT connection.
        """
        self.broker = broker
        self.port = port
        self.client_id = client_id
        self.client = mqtt.Client(client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the broker."""
        print(f"Connected to MQTT broker {self.broker} with result code {rc}")

    def on_message(self, client, userdata, message):
        """Callback for when a message is received."""
        print(f"Message received: {message.topic} {message.payload.decode()}")

    def connect(self):
        """Connect to the MQTT broker."""
        self.client.connect(self.broker, self.port)
        self.client.loop_start()

    def publish(self, topic, payload):
        """Publish a message to a topic.

        Args:
            topic (str): The topic to publish to.
            payload (str): The message payload.
        """
        self.client.publish(topic, payload)

    def subscribe(self, topic):
        """Subscribe to a topic.

        Args:
            topic (str): The topic to subscribe to.
        """
        self.client.subscribe(topic)
        self.client.on_message = self.on_message

    def disconnect(self):
        """Disconnect from the MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()
        print(f"Disconnected from MQTT broker {self.broker}.")
