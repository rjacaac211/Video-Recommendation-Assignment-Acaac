import unittest
from recommendation_engine.content_based import ContentBasedRecommender

class TestContentBasedRecommender(unittest.TestCase):
    def setUp(self):
        # Initialize the recommender with test data
        self.recommender = ContentBasedRecommender("../data/processed/all_posts_with_features.csv")
        self.recommender.compute_similarity()  # Ensure similarity matrix is computed

    def test_recommendations(self):
        post_id = 11  # Example post ID, replace with an actual ID
        recommendations = self.recommender.get_similar_posts(post_id, top_n=10)
        
        # Ensure the recommendations list is of length 10
        self.assertEqual(len(recommendations), 10)
        
        # Ensure all recommendations are valid post IDs
        self.assertTrue(all(isinstance(post_id, int) for post_id in recommendations))

if __name__ == "__main__":
    unittest.main()
