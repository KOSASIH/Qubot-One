# src/utils/data_processing.py

import pandas as pd

class DataProcessor:
    @staticmethod
    def clean_data(dataframe):
        """Clean the DataFrame by removing NaN values.

        Args:
            dataframe (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        cleaned_df = dataframe.dropna()
        print("Data cleaned. NaN values removed.")
        return cleaned_df

    @staticmethod
    def normalize_data(dataframe):
        """Normalize the DataFrame values.

        Args:
            dataframe (pd.DataFrame): The DataFrame to normalize.

        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        normalized_df = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
        print("Data normalized.")
        return normalized_df
