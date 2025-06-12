import unittest
from fastapi.testclient import TestClient
from perfume_haven.app import app  # Assumes app.py is in perfume_haven/

class FastAPIAppTests(unittest.TestCase):
    """Unit tests for the Perfume Haven FastAPI application."""

    def setUp(self):
        """Set up the TestClient before each test."""
        self.client = TestClient(app)

    def test_home_page(self):
        """Test the home page returns 200 and contains the correct title."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200, "Home page should return 200 OK")
        self.assertEqual(response.headers["content-type"], "text/html; charset=utf-8")
        self.assertIn(
            "<title>Scent & Memoire - Luxury Fragrances</title>",
            response.text,
            "Home page should contain the correct title"
        )

    def test_search_page_valid_query(self):
        """Test the search endpoint with a valid query."""
        query = {"query": "Evoke for Him Ajmal for men"}
        response = self.client.post("/search", json=query)
        self.assertEqual(response.status_code, 200, "Search endpoint should return 200 OK")
        self.assertEqual(response.headers["content-type"], "application/json")
        
        data = response.json()
        self.assertIn("results", data, "Response should contain 'results' key")
        self.assertIsInstance(data["results"], list, "'results' should be a list")
        
        if data["results"]:
            # Check the structure of the first result
            result = data["results"][0]
            self.assertIn("title", result, "Result should have 'title' field")
            self.assertIn("designer", result, "Result should have 'designer' field")
            self.assertIn("price", result, "Result should have 'price' field")
            self.assertIn("rating", result, "Result should have 'rating' field")

    def test_search_page_empty_query(self):
        """Test the search endpoint with an empty query."""
        query = {"query": ""}
        response = self.client.post("/search", json=query)
        self.assertEqual(response.status_code, 200, "Empty query should return 200 OK")
        self.assertEqual(response.headers["content-type"], "application/json")
        data = response.json()
        self.assertIn("results", data, "Response should contain 'results' key")
        self.assertEqual(data["results"], [], "Empty query should return empty results")

    def test_search_page_invalid_json(self):
        """Test the search endpoint with invalid JSON."""
        response = self.client.post("/search", content="invalid json")
        self.assertEqual(response.status_code, 422, "Invalid JSON should return 422 Unprocessable Entity")
        self.assertEqual(response.headers["content-type"], "application/json")
        data = response.json()
        self.assertIn("detail", data, "Response should contain 'detail' error message")

    def test_suggestions_endpoint(self):
        """Test the suggestions endpoint with a valid query."""
        response = self.client.get("/suggestions?query=rose")
        self.assertEqual(response.status_code, 200, "Suggestions endpoint should return 200 OK")
        self.assertEqual(response.headers["content-type"], "application/json")
        data = response.json()
        self.assertIn("suggestions", data, "Response should contain 'suggestions' key")
        self.assertIsInstance(data["suggestions"], list, "'suggestions' should be a list")

if __name__ == '__main__':
    unittest.main()