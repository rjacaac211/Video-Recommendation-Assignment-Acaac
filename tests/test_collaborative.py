import unittest
from recommendation_engine.collaborative import CollaborativeFilteringRecommender

class TestCollaborativeRecommender(unittest.TestCase):
    def setUp(self):
        # Initialize the recommender with test data
        self.recommender = CollaborativeFilteringRecommender("../data/processed/interaction_df.csv")
        self.recommender.compute_similarity()  # Compute the similarity matrix

    def test_recommendations(self):
        user_id = 1  # Example user ID, replace with an actual ID
        recommendations = self.recommender.recommend_posts(user_id, top_n=10)
        
        # Ensure the recommendations list is of length 10
        self.assertEqual(len(recommendations), 10)
        
        # Ensure all recommendations are valid post IDs
        self.assertTrue(all(isinstance(post_id, int) for post_id in recommendations))

if __name__ == "__main__":
    unittest.main()
