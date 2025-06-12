import unittest
import ast
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TestPerfumeRecommendation(unittest.TestCase):
    """Test the perfume recommendation system."""

    @classmethod
    def setUpClass(cls):
        # Load data
        data_path = "notebooks/perfumes_dataset.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError("Dataset not found")
        cls.data = pd.read_csv(data_path)

        # Preprocess tags (from app.py)
        def collapse(L):
            if isinstance(L, str):
                L = ast.literal_eval(L)
            return [i.replace(" ", "") for i in L]

        cls.data["tags"] = (
            cls.data["notes"].apply(collapse) +
            cls.data["title"].apply(lambda x: x.split())  
        )
        cls.data["tags"] = cls.data["tags"].apply(lambda x: " ".join(x).lower())

        # Create vectorizer
        cls.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        cls.tfidf_matrix = cls.vectorizer.fit_transform(cls.data["tags"])

    def test_recommendation_performance(self):
        """Test recommendation performance with Precision@k."""
        test_cases = [
            {
                "query": "Reserve Exclusif Vivamor Parfums for women and men",
                "relevant_titles": [
                    "Tobacco Supreme Vivamor Parfums for women and men",
                    "Rouge Imperiale Vivamor Parfums for women and men",
                    "Cherry Prive Vivamor Parfums for women and men"
                ],
                "k": 2
            },
            {
                "query": "04 Violet Blossom Zara for women",
                "relevant_titles": [
                    "01 Red Vanilla Zara for women",
                    "05 Woman Gold Zara for women"
                ],
                "k": 2
            }
        ]

        for case in test_cases:
            query = case["query"]
            relevant_titles = case["relevant_titles"]
            k = case["k"]

            # Vectorize query
            query_tfidf = self.vectorizer.transform([query])
            # Compute similarity
            sim_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
            # Get top-k indices
            predictions = np.argsort(sim_scores)[::-1][:k]

            predicted_titles = self.data.iloc[predictions]["title"].tolist()
            print(f"Query: {query}, Predicted: {predicted_titles}, Scores: {sim_scores[predictions]}")

            relevant_count = sum(title in relevant_titles for title in predicted_titles)
            precision_at_k = relevant_count / k

            self.assertGreaterEqual(
                precision_at_k, 0.33,
                f"Precision@{k} for '{query}' should be >= 0.33, got {precision_at_k}"
            )

if __name__ == "__main__":
    unittest.main()