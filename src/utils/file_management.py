# src/utils/file_management.py

import os
import json

class FileManager:
    @staticmethod
    def save_to_json(data, filename):
        """Save data to a JSON file.

        Args:
            data (dict): Data to save.
            filename (str): Filename to save the data to.
        """
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Data saved to {filename}.")

    @staticmethod
    def load_from_json(filename):
        """Load data from a JSON file.

        Args:
            filename (str): Filename to load the data from.

        Returns:
            dict: Loaded data.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Data loaded from {filename}.")
        return data

    @staticmethod
    def create_directory(directory):
        """Create a directory if it does not exist.

        Args:
            directory (str): Directory path to create.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created.")
        else:
            print(f"Directory {directory} already exists.")
