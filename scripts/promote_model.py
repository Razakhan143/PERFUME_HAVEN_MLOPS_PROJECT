import os
import argparse
import logging
import mlflow
from mlflow.exceptions import MlflowException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def promote_model(model_name, dagshub_token=None, repo_owner="Razakhan143", repo_name="PERFUME_HAVEN_MLOPS_PROJECT"):
    """
    Promote the latest model version from Staging to Production in MLflow.

    Args:
        model_name (str): Name of the registered model.
        dagshub_token (str, optional): DagsHub authentication token. Defaults to CAPSTONE_TEST env variable.
        repo_owner (str): DagsHub repository owner. Defaults to "Razakhan143".
        repo_name (str): DagsHub repository name. Defaults to "PERFUME_HAVEN_MLOPS_PROJECT".
    """
    try:
        # Validate DagsHub token
        token = dagshub_token or os.getenv("CAPSTONE_TEST")
        if not token:
            raise EnvironmentError("CAPSTONE_TEST environment variable or dagshub_token is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        # Set up MLflow tracking URI
        dagshub_url = "https://dagshub.com"
        tracking_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")

        # Initialize MLflow client
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)

        # Check if model exists
        try:
            client.get_registered_model(model_name)
        except MlflowException as e:
            logger.error(f"Registered model '{model_name}' not found: {e}")
            raise ValueError(f"Model '{model_name}' does not exist in the registry")

        # Get the latest version in Staging
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            logger.error("No model versions found in Staging stage")
            raise ValueError(f"No versions found for model '{model_name}' in Staging stage")

        latest_version = staging_versions[0].version
        logger.info(f"Latest Staging version: {latest_version}")

        # Promote the Staging version to Production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True  # Automatically archive existing Production versions
        )
        logger.info(f"Model version {latest_version} promoted to Production")

        # Verify the promotion
        model_version = client.get_model_version(model_name, latest_version)
        if model_version.current_stage == "Production":
            logger.info(f"Verified: Version {latest_version} is now in Production stage")
        else:
            logger.warning(f"Version {latest_version} is in stage: {model_version.current_stage}")

    except MlflowException as e:
        logger.error(f"MLflow error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Promote an MLflow model from Staging to Production")
    parser.add_argument("--model-name", type=str, default="my_model", help="Name of the registered model")
    parser.add_argument("--repo-owner", type=str, default="Razakhan143", help="DagsHub repository owner")
    parser.add_argument("--repo-name", type=str, default="PERFUME_HAVEN_MLOPS_PROJECT", help="DagsHub repository name")
    args = parser.parse_args()

    promote_model(
        model_name=args.model_name,
        repo_owner=args.repo_owner,
        repo_name=args.repo_name
    )

if __name__ == "__main__":
    main()