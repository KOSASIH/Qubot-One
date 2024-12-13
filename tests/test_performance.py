# tests/test_performance.py

import unittest
import time
from src.middleware import LoadBalancer  # Assuming you have a LoadBalancer class

class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.load_balancer = LoadBalancer()
        for i in range(5):  # Simulate adding servers
            self.load_balancer.add_server(f'http://localhost:{5000 + i}')

    def test_load_balancer_performance(self):
        start_time = time.time()
        for _ in range(1000):  # Simulate 1000 requests
            server = self.load_balancer.get_server()
        end_time = time.time()
        duration = end_time - start_time
        self.assertLess(duration, 1)  # Ensure it takes less than 1 second

if __name__ == '__main__':
    unittest.main()
