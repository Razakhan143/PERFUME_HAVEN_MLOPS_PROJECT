# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import sys
import os
import yaml
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logger import logging
from src.connections import s3_connection


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url, encoding='latin-1',on_bad_lines='skip')
        logging.info('Data loaded from %s', data_url)
        return df   
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # df.drop(columns=['tweet_id'], inplace=True)
        logging.info("pre-processing...")
        df.dropna( inplace=True)
        df.drop_duplicates(inplace=True)
        print(df.shape)
        logging.info('Data preprocessing completed')
        return df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(model_data: pd.DataFrame, data_path: str) -> None:
    """Save the model dataset."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        model_data.to_csv(os.path.join(raw_data_path, "model_data.csv"), index=False)
        logging.debug('Model data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        data_size = params['data_ingestion']['data_size']

        df = load_data('notebooks/perfumes_dataset.csv')

      #aws keys will be written in the src/connections/s3_connection.py file


        final_df = preprocess_data(df)
        model_data = final_df.sample(n=data_size, random_state=42)
        save_data(model_data, data_path='data')
        logging.info('Data ingestion process completed successfully')
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()