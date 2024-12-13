# scripts/data_collection.py

import json
import random
import time

def collect_data(num_samples):
    """Collect data samples for training and analysis."""
    data = []
    
    for _ in range(num_samples):
        sample = {
            'timestamp': time.time(),
            'sensor_value': random.uniform(0, 100),  # Simulate sensor data
            'status': random.choice(['normal', 'warning', 'critical'])
        }
        data.append(sample)
        print(f"Collected Data Sample: {sample}")
        time.sleep(1)  # Simulate time delay between samples

    # Save collected data to a JSON file
    with open('collected_data.json', 'w') as f:
        json.dump(data, f, indent=4)
    print("Data collection complete. Data saved to 'collected_data.json'.")

if __name__ == "__main__":
    collect_data(10)  # Collect 10 samples
