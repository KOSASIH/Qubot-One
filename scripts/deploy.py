# scripts/deploy.py

import os
import subprocess
import argparse

def deploy_application(environment):
    """Deploy the application to the specified environment."""
    print(f"Starting deployment to {environment}...")

    # Example deployment commands
    if environment == 'production':
        command = "git pull origin main && docker-compose up -d"
    elif environment == 'staging':
        command = "git pull origin staging && docker-compose up -d"
    else:
        raise ValueError("Invalid environment specified. Use 'production' or 'staging'.")

    # Execute the deployment command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print("Deployment successful!")
        print(stdout.decode())
    else:
        print("Deployment failed!")
        print(stderr.decode())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deployment script for the application.')
    parser.add_argument('environment', type=str, help='Deployment environment (production/staging)')
    args = parser.parse_args()
    
    deploy_application(args.environment)
