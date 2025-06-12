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
        # Load data with same sampling as main app
        data_path = "notebooks/perfumes_dataset.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError("Dataset not found")
        
        size_dataset = 10000  # Same as main app
        cls.data = pd.read_csv(data_path).sample(size_dataset, random_state=42).reset_index(drop=True)

        # Preprocess tags using same functions as main app
        def collapse(L):
            L1 = []
            for i in ast.literal_eval(L):
                L1.append(i.replace(" ",""))
            return L1

        cls.data['notes'] = cls.data['notes'].apply(collapse)
        cls.data['description'] = cls.data['description'].apply(lambda x: x.split())
        cls.data['designer'] = cls.data['designer'].apply(lambda x: x.split())
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

            # Find query index (same approach as main app)
            selected_indices = self.data[self.data['title'].isin([query])].index
            
            if len(selected_indices) > 0:
                # Calculate mean similarity (same as main app)
                sim_scores = cosine_similarity(
                    self.tfidf_matrix[selected_indices],
                    self.tfidf_matrix
                ).mean(axis=0)
            else:
                sim_scores = np.zeros(self.tfidf_matrix.shape[0])

            # Get top-k indices (excluding the query itself)
            similar_indices = sim_scores.argsort()[::-1]
            predicted_indices = [i for i in similar_indices if i not in selected_indices][:k]
            
            predicted_titles = self.data.iloc[predicted_indices]["title"].tolist()
            print(f"Query: {query}, Predicted: {predicted_titles}, Scores: {sim_scores[predicted_indices]}")

            # Calculate precision@k
            relevant_count = sum(title in relevant_titles for title in predicted_titles)
            precision_at_k = relevant_count / k

            self.assertGreaterEqual(
                precision_at_k, 0.33,
                f"Precision@{k} for '{query}' should be >= 0.33, got {precision_at_k}"
            )

if __name__ == "__main__":
    unittest.main()