
import unittest
import mlflow
import os
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer

class TestModelPerformance(unittest.TestCase):
    """Test the perfume recommendation model's performance."""

    @classmethod
    def setUpClass(cls):
        # Set DagsHub credentials
        token = os.getenv("CAPSTONE_TEST")
        if not token:
            raise EnvironmentError("CAPSTONE_TEST not set")
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        # Set MLflow tracking URI
        mlflow.set_tracking_uri("https://dagshub.com/Razakhan143/PERFUME_HAVEN_MLOPS_PROJECT.mlflow")

        # Load model
        model_name = "my_model"
        client = mlflow.MlflowClient()
        version = client.get_latest_versions(model_name, stages=["Staging"])[0].version
        cls.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")

        # Load data
        data_path = "notebooks/perfumes_dataset.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError("Dataset not found")
        cls.test_data = pd.read_csv(data_path)

        # Preprocess tags (mimic create_tags from app.py)
        def collapse(L):
            if isinstance(L, str):
                L = ast.literal_eval(L)
            return [i.replace(" ", "") for i in L]

        cls.test_data["tags"] = (
            cls.test_data["notes"].apply(collapse) +
            cls.test_data["designer"].apply(lambda x: x.split()) +
            cls.test_data["title"].apply(lambda x: x.split()) +
            cls.test_data["description"].apply(lambda x: x.split())
        )
        cls.test_data["tags"] = cls.test_data["tags"].apply(lambda x: " ".join(x).lower())

        # Create vectorizer (mimic app.py)
        cls.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        cls.X_test = cls.vectorizer.fit_transform(cls.test_data["tags"])

    def test_model_performance(self):
        """Test recommendation performance with Precision@k."""
        test_cases = [
            {
                "query": "rose perfume",
                "relevant_titles": [
                    "Sidra Nasamat Najd",
                    "Oud Plata Nasamat Najd",
                    "Sunset Eau de Parfum Intense Jil Sander"
                ],
                "k": 3
            },
            {
                "query": "floral fragrance",
                "relevant_titles": [
                    "Íris Avatim",
                    "Bambolê Tuberosa Louca"
                ],
                "k": 3
            }
        ]

        for case in test_cases:
            query = case["query"]
            relevant_titles = case["relevant_titles"]
            k = case["k"]

            query_tfidf = self.vectorizer.transform([query])
            input_df = pd.DataFrame(query_tfidf.toarray())

            try:
                predictions = self.model.predict(input_df)[:k]
            except Exception as e:
                self.fail(f"Prediction failed for '{query}': {e}")

            predicted_titles = self.test_data.iloc[predictions]["title"].tolist()
            relevant_count = sum(title in relevant_titles for title in predicted_titles)
            precision_at_k = relevant_count / k

            self.assertGreaterEqual(
                precision_at_k, 0.5,
                f"Precision@{k} for '{query}' should be >= 0.5, got {precision_at_k}"
            )

if __name__ == "__main__":
    unittest.main()