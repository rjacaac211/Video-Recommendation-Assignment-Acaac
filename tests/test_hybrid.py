import unittest
from recommendation_engine.content_based import ContentBasedRecommender
from recommendation_engine.collaborative import CollaborativeFilteringRecommender
from recommendation_engine.hybrid import HybridRecommender

class TestHybridRecommender(unittest.TestCase):
    def setUp(self):
        # Initialize the models
        content_model = ContentBasedRecommender("../data/processed/all_posts_with_features.csv")
        collaborative_model = CollaborativeFilteringRecommender("../data/processed/interaction_df.csv")

        # Compute necessary similarity matrices
        content_model.compute_similarity()
        collaborative_model.compute_similarity()

        # Initialize Hybrid Recommender
        self.hybrid_model = HybridRecommender(content_model, collaborative_model)

    def test_recommendations(self):
        user_id = 1  # Example user ID, replace with an actual ID
        recommendations = self.hybrid_model.recommend_posts(user_id, top_n=10)
        
        # Ensure the recommendations list is of length 10
        self.assertEqual(len(recommendations), 10)
        
        # Ensure all recommendations are valid post IDs
        self.assertTrue(all(isinstance(post_id, int) for post_id in recommendations))

if __name__ == "__main__":
    unittest.main()
