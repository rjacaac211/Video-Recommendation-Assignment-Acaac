import unittest
import requests

BASE_URL = "http://localhost:5000"  # Update if using a different port

class TestAPIEndpoints(unittest.TestCase):
    def test_get_recommendations(self):
        # Test for a valid user, category, and mood
        response = requests.get(f"{BASE_URL}/feed", params={"username": "user1", "category_id": 1, "mood": "happy"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recommendations", data)
        self.assertGreater(len(data["recommendations"]), 0)

    def test_missing_username(self):
        # Test for missing username parameter
        response = requests.get(f"{BASE_URL}/feed", params={"category_id": 1})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

if __name__ == "__main__":
    unittest.main()
