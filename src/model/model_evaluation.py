import numpy as np
import pandas as pd
import pickle
import json
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Import custom logger
from src.logger import logging


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "https://dagshub.com/Razakhan143/PERFUME_HAVEN_MLOPS_PROJECT.mlflow"
dagshub.init(repo_owner="Razakhan143", repo_name="PERFUME_HAVEN_MLOPS_PROJECT", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("capstone-trial")
# -------------------------------------------------------------------------------------

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise


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

def evaluate_model(cosine_sim, new_perfume, selected_perfumes, n=10) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        logging.info('Starting model evaluation')
        selected_indices = new_perfume[new_perfume['title'].isin(selected_perfumes)].index
        logging.info('Selected indices for evaluation: %s', selected_indices)

        sim_scores = cosine_sim[selected_indices].mean(axis=0)
        logging.info('Similarity scores calculated: %s', sim_scores.tolist())
        # Sort the indices based on similarity scores
        similar_indices = sim_scores.argsort()[::-1]
        logging.info('Similar indices calculated: %s', similar_indices.tolist())
        recommended_indices = [i for i in similar_indices if i not in selected_indices][:n]
        recommended_perfumes = new_perfume.iloc[recommended_indices][['title', 'designer', 'description', 'notes', 'img_url']]
    
        metrics_dict = {
            'recommended_perfumes': {recommended_perfumes.iloc[i]['title']: {
                'designer': recommended_perfumes.iloc[i]['designer'],
                'description': ' '.join(recommended_perfumes.iloc[i]['description']),
                'notes': ' '.join(recommended_perfumes.iloc[i]['notes']),
                'img_url': recommended_perfumes.iloc[i]['img_url']
            } for i in range(len(recommended_perfumes))}
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict, recommended_perfumes
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = load_model('model/cosine_similarity.pkl')
            test_data = load_data('data/interim/model_processed.csv')
            logging.info('Model and test data loaded successfully')
            selected_perfumes = test_data.sample(1)['title']
            print('Selected Perfumes: ' + ', '.join(selected_perfumes) + '\n')
            logging.info('Selected perfumes: %s', ', '.join(selected_perfumes))
            metrics, recommended_perfumes = evaluate_model(clf, test_data, selected_perfumes, 5)
            print('Recommended Perfumes:\n')
            logging.info('Recommended perfumes: %s', recommended_perfumes)
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            #having this error here root - ERROR - Failed to complete the model evaluation process: must be real number, not str Error: must be real number, not str
            mlflow.log_param("selected_perfumes", ', '.join(selected_perfumes))
            logging.info('Selected perfumes logged to MLflow')
            # Log the recommended perfumes to MLflow
            mlflow.log_param("recommended_perfumes", json.dumps(metrics['recommended_perfumes']))
            logging.info('Recommended perfumes logged to MLflow')
            # Log the model to MLflow
            mlflow.sklearn.log_model(clf, "cosine_similarity_model")
            logging.info('Model logged to MLflow')
            # Log the model run ID to MLflow
            mlflow.log_param("run_id", run.info.run_id)
            logging.info('Run ID logged to MLflow')
            # Log the model type to MLflow
            mlflow.log_param("model_type", "cosine_similarity")
            logging.info('Model type logged to MLflow')
            # Log the model name to MLflow
            mlflow.log_param("model_name", "cosine_similarity_model")
            logging.info('Model name logged to MLflow')
            # Log the model version to MLflow
            mlflow.log_param("model_version", "1.0")
            logging.info('Model version logged to MLflow')
            # Log the model description to MLflow
            mlflow.log_param("model_description", "Cosine similarity model for perfume recommendations")
            logging.info('Model description logged to MLflow')
            # Log the model creation date to MLflow
            mlflow.log_param("model_creation_date", pd.Timestamp.now().isoformat())
            logging.info('Model creation date logged to MLflow')
            # Log the model evaluation date to MLflow
            mlflow.log_param("model_evaluation_date", pd.Timestamp.now().isoformat())
            logging.info('Model evaluation date logged to MLflow')
            # Log the model evaluation time to MLflow
            mlflow.log_param("model_evaluation_time", pd.Timestamp.now().isoformat())
            logging.info('Model evaluation time logged to MLflow')
            # Log the model evaluation metrics to MLflow
            mlflow.log_param("model_evaluation_metrics", json.dumps(metrics))
            logging.info('Model evaluation metrics logged to MLflow')
            # Log the model evaluation parameters to MLflow
            mlflow.log_param("model_evaluation_parameters", json.dumps({'n': 5}))
            logging.info('Model evaluation parameters logged to MLflow')
            # Log the model evaluation results to MLflow
            mlflow.log_param("model_evaluation_results", json.dumps(metrics['recommended_perfumes']))
            logging.info('Model evaluation results logged to MLflow')
            # Log model parameters to MLflow
            mlflow.log_param("model_parameters", json.dumps({'n': 5}))
            logging.info('Model parameters logged to MLflow')
            # Log the model info to MLflow
            save_model_info(run.info.run_id, 'model/cosine_similarity.pkl', 'reports/model_info.json')
            logging.info('Model info saved to MLflow')  
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
