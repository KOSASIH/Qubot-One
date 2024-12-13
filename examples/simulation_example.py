# examples/simulation_example.py

import random
import time

def run_simulation():
    print("Starting simulation...")
    for i in range(10):
        time.sleep(1)  # Simulate time delay
        data_point = random.random()  # Simulate data generation
        print(f"Data Point {i + 1}: {data_point}")

if __name__ == "__main__":
    run_simulation()
