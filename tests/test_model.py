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
        # Load full data without sampling to ensure test perfumes exist
        data_path = "notebooks/perfumes_dataset.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError("Dataset not found")
        
        cls.data = pd.read_csv(data_path)

        # Preprocess tags using same functions as main app
        def collapse(L):
            if isinstance(L, str):
                L = ast.literal_eval(L)
            return [i.replace(" ","") for i in L]

        cls.data['notes'] = cls.data['notes'].apply(collapse)
        cls.data['description'] = cls.data['description'].apply(lambda x: x.split() if isinstance(x, str) else [])
        cls.data['designer'] = cls.data['designer'].apply(lambda x: x.split() if isinstance(x, str) else [])
        cls.data['title_split'] = cls.data['title'].apply(lambda x: x.split())
        
        cls.data["tags"] = (
            cls.data['notes'] +
            cls.data['designer'] +
            cls.data['title_split'] +
            cls.data['description']
        )
        cls.data["tags"] = cls.data["tags"].apply(lambda x: " ".join(x)).str.lower()

        # Create vectorizer with same parameters as main app
        cls.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        cls.tfidf_matrix = cls.vectorizer.fit_transform(cls.data["tags"])

    def test_recommendation_performance(self):
        """Test recommendation performance with Precision@k."""
        test_cases = [
            {
                "query": "Black Opium",  # More common perfume that likely exists
                "relevant_titles": [
                    "Black Opium Illicit Green",
                    "Black Opium Neon",
                    "Black Opium Extreme"
                ],
                "k": 3
            },
            {
                "query": "Sauvage",  # Another common perfume line
                "relevant_titles": [
                    "Sauvage Eau de Parfum",
                    "Sauvage Elixir",
                    "Sauvage Parfum"
                ],
                "k": 3
            }
        ]

        for case in test_cases:
            query = case["query"]
            relevant_titles = case["relevant_titles"]
            k = case["k"]

            # Find perfumes containing the query string
            query_mask = self.data['title'].str.contains(query, case=False)
            selected_indices = self.data[query_mask].index
            
            if len(selected_indices) == 0:
                print(f"Warning: No perfumes found matching query '{query}'")
                continue

            # Calculate mean similarity (same as main app)
            sim_scores = cosine_similarity(
                self.tfidf_matrix[selected_indices],
                self.tfidf_matrix
            ).mean(axis=0)

            # Get top-k indices (excluding the query itself)
            similar_indices = sim_scores.argsort()[::-1]
            predicted_indices = [i for i in similar_indices if i not in selected_indices][:k]
            
            predicted_titles = self.data.iloc[predicted_indices]["title"].tolist()
            print(f"\nQuery: {query}")
            print(f"Predicted: {predicted_titles}")
            print(f"Scores: {sim_scores[predicted_indices]}")

            # Calculate precision@k
            relevant_count = sum(any(rt in title for rt in relevant_titles) for title in predicted_titles)
            precision_at_k = relevant_count / k

            self.assertGreaterEqual(
                precision_at_k, 0.33,
                f"Precision@{k} for '{query}' should be >= 0.33, got {precision_at_k}"
            )

if __name__ == "__main__":
    unittest.main()