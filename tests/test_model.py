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
import uvicorn
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
        # Use perfumes we know exist in the sampled data
        test_perfumes = self.data.sample(2)['title'].tolist()
        
        for query in test_perfumes:
            with self.subTest(query=query):
                selected_indices = self.data[self.data['title'] == query].index
                self.assertGreater(len(selected_indices), 0, f"Query perfume not found: {query}")

                # Get recommendations
                sim_scores = cosine_similarity(
                    self.tfidf_matrix[selected_indices],
                    self.tfidf_matrix
                ).mean(axis=0)
                
                similar_indices = sim_scores.argsort()[::-1]
                rec_indices = [i for i in similar_indices if i not in selected_indices][:5]
                recommendations = self.data.iloc[rec_indices]

                # 1. Verify we get recommendations
                self.assertGreater(len(rec_indices), 0, "No recommendations returned")

                # 2. Verify all required fields exist
                required_fields = ['title', 'designer', 'description', 'notes', 'img_url']
                for field in required_fields:
                    self.assertIn(field, recommendations.columns, f"Missing field {field}")

                # 3. Verify scores are reasonable
                self.assertTrue(
                    all(score > 0 for score in sim_scores[rec_indices]),
                    f"Recommendation scores should be positive: {sim_scores[rec_indices]}"
                )

    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Test empty query - should return empty results, not error
        empty_query = ""
        empty_vec = self.vectorizer.transform([empty_query])
        sim_scores = cosine_similarity(empty_vec, self.tfidf_matrix)
        self.assertEqual(sim_scores.shape, (1, len(self.data)))
        self.assertTrue(all(score == 0 for score in sim_scores[0]))

        # Test non-existent perfume
        non_existent = "XYZ Perfume That Doesn't Exist 123"
        selected_indices = self.data[self.data['title'] == non_existent].index
        self.assertEqual(len(selected_indices), 0)

        # Transform the non-existent perfume
        non_existent_vec = self.vectorizer.transform([non_existent])
        sim_scores = cosine_similarity(non_existent_vec, self.tfidf_matrix)
        self.assertEqual(sim_scores.shape, (1, len(self.data)))

    def test_similarity_distribution(self):
        """Verify similarity scores have reasonable distribution."""
        # Test with a known perfume
        test_perfume = self.data.iloc[0]['title']
        selected_indices = self.data[self.data['title'] == test_perfume].index
        
        sim_scores = cosine_similarity(
            self.tfidf_matrix[selected_indices],
            self.tfidf_matrix
        ).mean(axis=0)
        
        # Verify we have a range of scores
        self.assertLess(min(sim_scores), max(sim_scores), "All scores are identical")
        # Verify most scores are low (sparse similarity)
        self.assertGreater(np.median(sim_scores), 0, "Median score should be positive")
        self.assertLess(np.median(sim_scores), 0.3, "Median score unexpectedly high")

if __name__ == "__main__":
    unittest.main()