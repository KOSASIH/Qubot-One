# src/middleware/event_bus.py

class EventBus:
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers = {}

    def subscribe(self, event_type, callback):
        """Subscribe to an event type.

        Args:
            event_type (str): The type of event to subscribe to.
            callback (callable): The callback function to handle the event.
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        print(f"Subscribed to event: {event_type}")

    def publish(self, event_type, data):
        """Publish an event to all subscribers.

        Args:
            event_type (str): The type of event to publish.
            data (dict): The data associated with the event.
        """
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(data)
            print(f"Published event: {event_type} with data: {data}")
        else:
            print(f"No subscribers for event: {event_type}")
