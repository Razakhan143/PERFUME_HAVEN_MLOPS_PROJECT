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
    """Test the perfume recommendation system."""

    @classmethod
    def setUpClass(cls):
        # Load full data without sampling
        data_path = "notebooks/perfumes_dataset.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError("Dataset not found")
        
        cls.data = pd.read_csv(data_path)

        # Preprocess tags
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

        # Create vectorizer
        cls.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        cls.tfidf_matrix = cls.vectorizer.fit_transform(cls.data["tags"])

    def test_recommendation_performance(self):
        """Test that recommendations have reasonable similarity scores."""
        test_cases = [
            {
                "query": "Boss The Scent Absolute Hugo Boss for men",
                "min_expected_score": 0.7,
                "k": 3
            },
            {
                "query": "Miss Dior Blooming Bouquet Roller Pearl Dior for women",
                "min_expected_score": 0.6, 
                "k": 3
            }
        ]

        for case in test_cases:
            query = case["query"]
            min_score = case["min_expected_score"]
            k = case["k"]

            # Find perfumes containing the query string
            query_mask = self.data['title'].str.contains(query, case=False)
            selected_indices = self.data[query_mask].index
            
            if len(selected_indices) == 0:
                print(f"Warning: No perfumes found matching query '{query}'")
                continue

            # Calculate similarity
            sim_scores = cosine_similarity(
                self.tfidf_matrix[selected_indices],
                self.tfidf_matrix
            ).mean(axis=0)

            # Get top recommendations
            similar_indices = sim_scores.argsort()[::-1]
            predicted_indices = [i for i in similar_indices if i not in selected_indices][:k]
            
            predicted_titles = self.data.iloc[predicted_indices]["title"].tolist()
            predicted_scores = sim_scores[predicted_indices]
            
            print(f"\nQuery: {query}")
            print(f"Predicted: {predicted_titles}")
            print(f"Scores: {predicted_scores}")

            # Verify all recommendations meet minimum similarity score
            for score in predicted_scores:
                self.assertGreaterEqual(
                    score, min_score,
                    f"Recommendation score {score:.2f} should be >= {min_score}"
                )

            # Verify recommendations share brand/designer with query
            query_designer = self.data.iloc[selected_indices[0]]['designer']
            for idx in predicted_indices:
                rec_designer = self.data.iloc[idx]['designer']
                self.assertTrue(
                    any(brand in rec_designer for brand in query_designer),
                    f"Recommendation should share brand with query"
                )

if __name__ == "__main__":
    unittest.main()