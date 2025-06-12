# data preprocessing

import numpy as np
import pandas as pd
import os
import re
import ast
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logger import logging

def preprocess_dataframe(df):
    """
    Preprocess a DataFrame by applying text preprocessing to a specific column.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Initialize lemmatizer and stopwords
    def collapse(L):
        try:
            L1 = []
            for i in ast.literal_eval(L):
                L1.append(i.replace(" ", ""))
            return L1
        except Exception as e:
            print(f"Error in collapse: {e}")
            return []

    def normalize_text(perfumes):
        try:
            perfumes = perfumes.copy()
            perfumes['notes'] = perfumes['notes'].apply(collapse)
            perfumes['description'] = perfumes['description'].apply(lambda x: x.split() if isinstance(x, str) else [])
            perfumes['designer'] = perfumes['designer'].apply(lambda x: x.split() if isinstance(x, str) else [])
            perfumes['tags'] = perfumes['notes'] + perfumes['description'] + perfumes['designer']
            perfumes['tags'] = perfumes['tags'].apply(lambda x: " ".join([str(i) for i in x if i])).str.lower()
            perfumes['notes'] = perfumes['notes'].apply(
                lambda x: ', '.join(word.title() for word in x) if isinstance(x, list) else str(x).capitalize()
            )
            perfumes.reset_index(drop=True, inplace=True)
            return perfumes
        except Exception as e:
            print(f"Error during text normalization: {e}")
            raise

    logging.info("Starting data pre-processing...")
    # Apply the normalization function to the DataFrame
    df = normalize_text(df)
    logging.info("Data pre-processing completed")
    return df


def main():
    try:
        # Fetch the data from data/raw
        model_data = pd.read_csv('data/raw/model_data.csv')
        logging.info('data loaded properly')

        # Transform the data
        model_processed_data = preprocess_dataframe(model_data)

        # Store the data inside data/processed
        data_path = os.path.join("data", "interim")
        os.makedirs(data_path, exist_ok=True)

        model_processed_data.to_csv(os.path.join(data_path, "model_processed.csv"), index=False)

        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()