# src/middleware/message_broker.py

import pika
import json

class MessageBroker:
    def __init__(self, host='localhost'):
        """Initialize the message broker.

        Args:
            host (str): Hostname of the message broker.
        """
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host))
        self.channel = self.connection.channel()
        print("Message broker connected.")

    def publish(self, exchange, routing_key, message):
        """Publish a message to a specified exchange.

        Args:
            exchange (str): The exchange to publish to.
            routing_key (str): The routing key for the message.
            message (dict): The message to send.
        """
        self.channel.exchange_declare(exchange=exchange, exchange_type='topic')
        self.channel.basic_publish(exchange=exchange, routing_key=routing_key, body=json.dumps(message))
        print(f"Published message to {exchange} with routing key {routing_key}: {message}")

    def subscribe(self, queue, callback):
        """Subscribe to a queue and set a callback for message processing.

        Args:
            queue (str): The queue to subscribe to.
            callback (callable): The callback function to process messages.
        """
        self.channel.queue_declare(queue=queue)
        self.channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)
        print(f"Subscribed to queue: {queue}")
        self.channel.start_consuming()

    def close(self):
        """Close the message broker connection."""
        self.connection.close()
        print("Message broker connection closed.")
