import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from content_based import ContentBasedRecommender
from collaborative import CollaborativeFilteringRecommender


class HybridRecommender:
    def __init__(self, content_model, collaborative_model, weight_content=0.3, weight_collaborative=0.7):
        """
        Initialize the Hybrid Recommender.
        Args:
            content_model (ContentBasedRecommender): An instance of Content-Based Recommender.
            collaborative_model (CollaborativeFilteringRecommender): An instance of Collaborative Filtering Recommender.
            weight_content (float): Weight for content-based recommendations.
            weight_collaborative (float): Weight for collaborative recommendations.
        """
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.weight_content = weight_content
        self.weight_collaborative = weight_collaborative

    def recommend_posts(self, user_id, top_n=10):
        """
        Generate recommendations using a hybrid approach.
        Args:
            user_id (int): The ID of the user to generate recommendations for.
            top_n (int): Number of posts to recommend.
        Returns:
            pd.DataFrame: A DataFrame containing top-N recommended post IDs and scores.
        """
        # Get predictions from content-based and collaborative models
        content_scores = self.content_model.predict_scores(user_id)
        collaborative_scores = self.collaborative_model.predict_scores(user_id)

        # Normalize scores
        scaler = MinMaxScaler()
        normalized_content = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
        normalized_collaborative = scaler.fit_transform(collaborative_scores.reshape(-1, 1)).flatten()

        # Combine scores using weighted aggregation
        hybrid_scores = (
            self.weight_content * normalized_content +
            self.weight_collaborative * normalized_collaborative
        )

        # Sort posts by hybrid scores
        post_ids = self.content_model.get_post_ids()
        recommendations = pd.DataFrame({'post_id': post_ids, 'score': hybrid_scores})
        recommendations = recommendations.sort_values(by='score', ascending=False).head(top_n)

        return recommendations


# Example Usage
if __name__ == "__main__":
    # Initialize individual models
    content_model = ContentBasedRecommender("../data/processed/all_posts_with_features.csv")
    collaborative_model = CollaborativeFilteringRecommender("../data/processed/interaction_df.csv")

    # Build necessary matrices
    content_model.build_similarity_matrix()
    collaborative_model.compute_similarity()

    # Initialize Hybrid Recommender
    hybrid_model = HybridRecommender(content_model, collaborative_model)

    # Get recommendations for a user
    user_id = 1  # Replace with a valid user ID
    recommendations = hybrid_model.recommend_posts(user_id, top_n=10)
    print(f"Hybrid Recommendations for User {user_id}:")
    print(recommendations)
