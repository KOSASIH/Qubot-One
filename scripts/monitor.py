# scripts/monitor.py

import time
import random
import logging

# Configure logging
logging.basicConfig(filename='qubot_monitor.log', level=logging.INFO)

def monitor_qubots():
    """Monitor Qubots and log their performance metrics."""
    while True:
        # Simulate Qubot status and performance metrics
        qubot_id = random.randint(1, 10)
        status = random.choice(['active', 'idle', 'error'])
        battery_level = random.randint(0, 100)
        
        # Log the metrics
        logging.info(f"Qubot ID: {qubot_id}, Status: {status}, Battery Level: {battery_level}%")
        print(f"Qubot ID: {qubot_id}, Status: {status}, Battery Level: {battery_level}%")
        
        time.sleep(5)  # Monitor every 5 seconds

if __name__ == "__main__":
    monitor_qubots()
