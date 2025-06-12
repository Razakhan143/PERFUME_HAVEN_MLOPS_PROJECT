import os
import yaml
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logger import logging

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

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        if 'tags' not in df.columns:
            raise ValueError("The 'tags' column is missing from the DataFrame.")
        if df['tags'].str.strip().eq('').all():
            raise ValueError("The 'tags' column contains only empty strings.")
        logging.info('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_vectorizer(df: pd.DataFrame, max_features: int) -> tuple:
    """Apply TF-IDF Vectorizer and optionally Cosine Similarity to the data."""
    try:
        logging.info("Applying TF-IDF vectorization...")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(df['tags'])
        logging.info("TF-IDF Vectorization completed with max_features=%d", max_features)
        
        logging.info("TF-IDF applied successfully.")
        return tfidf_matrix
    except Exception as e:
        logging.error('Error during TF-IDF application: %s', e)
        raise

def save_tfidf_matrix(tfidf_matrix: csr_matrix, file_path: str) -> None:
    """Save the TF-IDF matrix to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_npz(file_path, tfidf_matrix)
        logging.info('TF-IDF matrix saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the TF-IDF matrix: %s', e)
        raise

def main():
    try:
        # Load parameters from YAML
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        # Example value, can be replaced with a parameter from a config file
        logging.info('Starting feature engineering process...')
        # Load data
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
        model_data = load_data(os.path.join(data_dir, 'interim/model_processed.csv'))
        logging.info('Data loaded for feature engineering.')

        if model_data.empty:
            logging.error('The input data is empty. Please check the data source.')
            raise ValueError("Input data is empty.")

        # Apply TF-IDF vectorization
        tfidf_matrix = apply_vectorizer(model_data, max_features)
        logging.info('TF-IDF matrix created successfully.')

        # Save the TF-IDF matrix
        save_tfidf_matrix(tfidf_matrix, os.path.join(data_dir, 'processed/tfidf_matrix.npz'))

        logging.info('TF-IDF saved to %s', os.path.join(data_dir, 'processed/tfidf_matrix.npz'))

    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()