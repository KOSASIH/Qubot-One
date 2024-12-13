# src/utils/data_validation.py

class DataValidator:
    @staticmethod
    def validate_data(data, schema):
        """Validate data against a schema.

        Args:
            data (dict): Data to validate.
            schema (dict): Schema to validate against.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        for key, value in schema.items():
            if key not in data or not isinstance(data[key], value):
                print(f"Validation failed for key: {key}. Expected type: {value}.")
                return False
        print("Data validation passed.")
        return True
