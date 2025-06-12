# import unittest
# import ast
# import numpy as np
# import pandas as pd
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# class TestPerfumeRecommendation(unittest.TestCase):
#     """Test the perfume recommendation system."""

#     @classmethod
#     def setUpClass(cls):
#         # Load data with same sampling as main app
#         data_path = "notebooks/perfumes_dataset.csv"
#         if not os.path.exists(data_path):
#             raise FileNotFoundError("Dataset not found")
        
#         size_dataset = 10000  # Same as main app
#         cls.data = pd.read_csv(data_path).sample(size_dataset, random_state=42).reset_index(drop=True)

#         # Preprocess tags using same functions as main app
#         def collapse(L):
#             L1 = []
#             for i in ast.literal_eval(L):
#                 L1.append(i.replace(" ",""))
#             return L1

#         cls.data['notes'] = cls.data['notes'].apply(collapse)
#         cls.data['description'] = cls.data['description'].apply(lambda x: x.split())
#         cls.data['designer'] = cls.data['designer'].apply(lambda x: x.split())
#         cls.data['title_split'] = cls.data['title'].apply(lambda x: x.split())
        
#         cls.data["tags"] = (
#             cls.data['notes'] +
#             cls.data['designer'] +
#             cls.data['title_split'] +
#             cls.data['description']
#         )
#         cls.data["tags"] = cls.data["tags"].apply(lambda x: " ".join(x)).str.lower()

#         # Create vectorizer with same parameters as main app
#         cls.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
#         cls.tfidf_matrix = cls.vectorizer.fit_transform(cls.data["tags"])

#     def test_recommendation_performance(self):
#         """Test recommendation performance with Precision@k."""
#         test_cases = [
#             {
#                 "query": "Boss The Scent Absolute Hugo Boss for men",
#                 "relevant_titles": [
#                     "The Collection Confident Oud Hugo Boss for men",
#                     "Boss The Scent For Her Parfum Edition Hugo Boss for women",
#                     "Boss The Scent Absolute Hugo Boss for men"
#                 ],
#                 "k": 2
#             },
#             {
#                 "query": "Miss Dior Blooming Bouquet Roller Pearl Dior for women",
#                 "relevant_titles": [
#                     "Miss Dior Blooming Bouquet Roller Pearl Dior for women",
#                     "Patchouli Imperial Dior for women and men"
#                 ],
#                 "k": 2
#             }
#         ]

#         for case in test_cases:
#             query = case["query"]
#             relevant_titles = case["relevant_titles"]
#             k = case["k"]

#             # Find query index (same approach as main app)
#             selected_indices = self.data[self.data['title'].isin([query])].index
            
#             if len(selected_indices) > 0:
#                 # Calculate mean similarity (same as main app)
#                 sim_scores = cosine_similarity(
#                     self.tfidf_matrix[selected_indices],
#                     self.tfidf_matrix
#                 ).mean(axis=0)
#             else:
#                 sim_scores = np.zeros(self.tfidf_matrix.shape[0])

#             # Get top-k indices (excluding the query itself)
#             similar_indices = sim_scores.argsort()[::-1]
#             predicted_indices = [i for i in similar_indices if i not in selected_indices][:k]
            
#             predicted_titles = self.data.iloc[predicted_indices]["title"].tolist()
#             print(f"Query: {query}, Predicted: {predicted_titles}, Scores: {sim_scores[predicted_indices]}")

#             # Calculate precision@k
#             relevant_count = sum(title in relevant_titles for title in predicted_titles)
#             precision_at_k = relevant_count / k

#             self.assertGreaterEqual(
#                 precision_at_k, 0.33,
#                 f"Precision@{k} for '{query}' should be >= 0.33, got {precision_at_k}"
#             )

# if __name__ == "__main__":
#     unittest.main()

import unittest
import ast
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TestPerfumeRecommendation(unittest.TestCase):
    """Comprehensive tests for the perfume recommendation system."""

    @classmethod
    def setUpClass(cls):
        # Load data EXACTLY like the main app does
        data_path = "notebooks/perfumes_dataset.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError("Dataset not found")
        
        # Same sampling as main app
        size_dataset = 10000
        cls.data = pd.read_csv(data_path).sample(size_dataset, random_state=42).reset_index(drop=True)

        # EXACT copy of the app's processing functions
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

        # Same vectorizer parameters as main app
        cls.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        cls.tfidf_matrix = cls.vectorizer.fit_transform(cls.data["tags"])

    def test_recommendation_quality(self):
        """Test that recommendations meet quality thresholds."""
        test_cases = [
            {
                "query": "Sauvage Dior for men",
                "min_score": 0.5,
                "required_fields": ['title', 'designer', 'description', 'notes', 'img_url'],
                "k": 5
            },
            {
                "query": "Black Opium YSL for women",
                "min_score": 0.5,
                "required_fields": ['title', 'designer', 'description', 'notes', 'img_url'],
                "k": 5
            }
        ]

        for case in test_cases:
            with self.subTest(query=case["query"]):
                self._validate_recommendation(case)

    def _validate_recommendation(self, case):
        query = case["query"]
        selected_indices = self.data[self.data['title'].str.contains(query, case=False)].index
        
        if len(selected_indices) == 0:
            self.skipTest(f"Query perfume not found in sample: {query}")

        # Get recommendations
        sim_scores = cosine_similarity(
            self.tfidf_matrix[selected_indices],
            self.tfidf_matrix
        ).mean(axis=0)
        
        similar_indices = sim_scores.argsort()[::-1]
        rec_indices = [i for i in similar_indices if i not in selected_indices][:case["k"]]
        recommendations = self.data.iloc[rec_indices]

        # 1. Verify scores meet minimum threshold
        self.assertTrue(
            all(sim_scores[rec_indices] >= case["min_score"]),
            f"Scores {sim_scores[rec_indices]} below threshold {case['min_score']}"
        )

        # 2. Verify all required fields exist
        for field in case["required_fields"]:
            self.assertIn(field, recommendations.columns, f"Missing field {field}")

        # 3. Verify at least some brand overlap
        query_brands = set(self.data.iloc[selected_indices[0]]['designer'])
        rec_brands = set().union(*recommendations['designer'].apply(set))
        self.assertTrue(
            query_brands & rec_brands,
            f"No brand overlap between query and recommendations"
        )

    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Empty query
        with self.assertRaises(ValueError):
            cosine_similarity(
                self.vectorizer.transform([""]),
                self.tfidf_matrix
            )

        # Non-existent perfume
        non_existent = "XYZ Perfume That Doesn't Exist"
        selected_indices = self.data[self.data['title'] == non_existent].index
        self.assertEqual(len(selected_indices), 0)

        # Verify it doesn't crash
        sim_scores = cosine_similarity(
            self.vectorizer.transform([non_existent]),
            self.tfidf_matrix
        )
        self.assertEqual(sim_scores.shape, (1, len(self.data)))

if __name__ == "__main__":
    unittest.main()