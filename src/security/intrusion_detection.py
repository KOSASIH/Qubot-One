# src/security/intrusion_detection.py

import time

class IntrusionDetection:
    def __init__(self):
        """Initialize the intrusion detection system."""
        self.intrusion_logs = []

    def log_intrusion_attempt(self, ip_address, timestamp=None):
        """Log an intrusion attempt.

        Args:
            ip_address (str): The IP address of the intrusion attempt.
            timestamp (float): The time of the attempt. Defaults to current time.
        """
        if timestamp is None:
            timestamp = time.time()
        self.intrusion_logs.append({'ip_address': ip_address, 'timestamp': timestamp})
        print(f"Intrusion attempt logged from IP: {ip_address} at {timestamp}")

    def get_intrusion_logs(self):
        """Get the list of intrusion logs.

        Returns:
            list: List of intrusion logs.
        """
        return self.intrusion_logs
