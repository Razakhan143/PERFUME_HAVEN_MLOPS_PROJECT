import unittest
from fastapi.testclient import TestClient
from perfume_haven.app import app
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FastAPIAppTests(unittest.TestCase):
    """Tests for the FastAPI application endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_root_page(self):
        """Test the root endpoint returns index.html."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
    
    def test_search_page_valid_query(self):
        """Test the search endpoint with a valid query."""
        response = self.client.post("/search", json={"query": "Evoke for Him Ajmal for men"})
        self.assertEqual(response.status_code, 200)
        
        results = response.json()["results"]
        self.assertGreater(len(results), 0, "Should return at least one result")
        
        # Check fields that actually exist in your response
        first_result = results[0]
        required_fields = ['title', 'designer', 'description', 'notes', 'img_url']
        for field in required_fields:
            self.assertIn(field, first_result, f"Result should have '{field}' field")
    
    def test_search_page_empty_query(self):
        """Test the search endpoint with an empty query."""
        response = self.client.post("/search", json={"query": ""})
        self.assertEqual(response.status_code, 400, "Empty query should return 400 Bad Request")
        self.assertIn("detail", response.json())
    
    def test_search_results_page(self):
        """Test the search results page loads."""
        response = self.client.get("/search-results.html")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
    
    def test_suggestions_endpoint(self):
        """Test the suggestions endpoint."""
        response = self.client.get("/suggestions", params={"query": "Evoke"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("suggestions", response.json())

if __name__ == "__main__":
    unittest.main()