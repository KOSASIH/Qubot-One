# tests/test_middleware.py

import unittest
from src.middleware import MessageBroker  # Assuming you have a MessageBroker class

class TestMiddleware(unittest.TestCase):
    def setUp(self):
        self.broker = MessageBroker()

    def test_publish_message(self):
        message = {'key': 'value'}
        self.broker.publish('test_exchange', 'test.routing.key', message)
        # Add assertions to verify message was published

    def test_subscribe_message(self):
        def callback(ch, method, properties, body):
            self.received_message = body

        self.broker.subscribe('test_queue', callback)
        # Add logic to publish a message and verify it was received

if __name__ == '__main__':
    unittest.main()
