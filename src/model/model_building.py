import numpy as np
import pandas as pd
import pickle
import yaml
import sys
import os
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def load_tfidf_matrix(file_path: str):
    """Load a TF-IDF matrix from a .npz file."""
    try:
        if not os.path.exists(file_path):
            logging.error('File not found: %s', file_path)
            raise FileNotFoundError(f"TF-IDF matrix file not found: {file_path}")
        tfidf_matrix = load_npz(file_path)
        logging.info('TF-IDF matrix loaded from %s', file_path)
        return tfidf_matrix
    except Exception as e:
        logging.error('Error loading TF-IDF matrix: %s', e)
        raise


def apply_cosine_similarity(df: pd.DataFrame, tfidf_matrix: csr_matrix) -> pd.DataFrame:
    """Apply recommendations based on the TF-IDF matrix."""
    try:
        # Compute cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        # Create a DataFrame for the recommendations
        cosine_sim = pd.DataFrame(cosine_sim)
        logging.info('Recommendations applied successfully')
        return cosine_sim
    except Exception as e:
        logging.error('Error during recommendations application: %s', e)
        raise

def save_cosine_similarity(model, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred “while saving the data: %s', e)
        raise

def main():
    try:

        interim_data = load_data('data/interim/model_processed.csv')
        tfidf_matrix = load_tfidf_matrix('data/processed/tfidf_matrix.npz')
        clf = apply_cosine_similarity(interim_data, tfidf_matrix)


        with open('model/cosine_similarity.pkl', 'wb') as file:
            pickle.dump(clf, file)
        logging.info('Model building process completed successfully')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()