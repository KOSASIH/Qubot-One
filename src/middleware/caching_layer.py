# src/middleware/caching_layer.py

import time

class CachingLayer:
    def __init__(self, expiration_time=60):
        """Initialize the caching layer.

        Args:
            expiration_time (int): Time in seconds before cache expires.
        """
        self.cache = {}
        self.expiration_time = expiration_time

    def set(self, key, value):
        """Set a value in the cache.

        Args:
            key (str): The cache key.
            value (any): The value to cache.
        """
        self.cache[key] = (value, time.time())
        print(f"Cached value for key: {key}")

    def get(self, key):
        """Get a value from the cache.

        Args:
            key (str): The cache key.

        Returns:
            any: The cached value or None if expired or not found.
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.expiration_time:
                print(f"Cache hit for key: {key}")
                return value
            else:
                del self.cache[key]  # Remove expired cache
                print(f"Cache expired for key: {key}")
        print(f"Cache miss for key: {key}")
        return None
