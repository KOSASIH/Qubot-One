# tests/test_utils.py

import unittest
from src.utils import DataProcessor  # Assuming you have a DataProcessor class

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()

    def test_clean_data(self):
        input_data = [1, 2, None, 4]
        cleaned_data = self.processor.clean_data(input_data)
        self.assertEqual(cleaned_data, [1, 2, 4])  # Adjust based on expected behavior

    def test_normalize_data(self):
        input_data = [1, 2, 3, 4]
        normalized_data = self.processor.normalize_data(input_data)
        self.assertEqual(normalized_data, [0, 0.33, 0.67, 1])  # Adjust based on expected behavior

if __name__ == '__main__':
    unittest.main()
