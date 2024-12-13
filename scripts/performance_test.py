# scripts/performance_test.py

import time
import requests

def performance_test(url, iterations):
    """Test the performance of the application by sending requests."""
    print(f"Starting performance test on {url} for {iterations} iterations...")
    
    total_time = 0
    for i in range(iterations):
        start_time = time.time()
        response = requests.get(url)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        
        print(f"Iteration {i + 1}: Response Code: {response.status_code}, Time Taken: {elapsed_time:.4f} seconds")
    
    average_time = total_time / iterations
    print(f"Average Response Time: {average_time:.4f} seconds")

if __name__ == "__main__":
    target_url = "http://localhost:5000"  # Change to your application's URL
    performance_test(target_url, 10)  # Test for 10 iterations
