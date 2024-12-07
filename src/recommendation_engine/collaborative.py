import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringRecommender:
    def __init__(self, interaction_data_path):
        """
        Initialize the Collaborative Filtering Recommender.
        Args:
            interaction_data_path (str): Path to the CSV file containing user-post interaction data.
        """
        self.interaction_df = pd.read_csv(interaction_data_path)
        self.user_post_matrix = None
        self.similarity_matrix = None

    def build_user_post_matrix(self):
        """
        Build the user-post interaction matrix.
        """
        self.user_post_matrix = self.interaction_df.pivot_table(
            index="user_id",
            columns="post_id",
            values="interaction_type",
            aggfunc=lambda x: 1 if len(x) > 0 else 0
        ).fillna(0)

    def compute_similarity(self):
        """
        Compute item-item similarity matrix based on the user-post interaction matrix.
        """
        if self.user_post_matrix is None:
            self.build_user_post_matrix()
        self.similarity_matrix = cosine_similarity(self.user_post_matrix.T)

    def recommend_posts(self, user_id, top_n=10):
        """
        Recommend posts for a given user based on collaborative filtering.
        Args:
            user_id (int): The ID of the user to generate recommendations for.
            top_n (int): Number of posts to recommend.
        Returns:
            list: List of recommended post IDs.
        """
        if self.similarity_matrix is None:
            self.compute_similarity()

        # Get the posts the user has interacted with
        user_interactions = self.user_post_matrix.loc[user_id]
        interacted_posts = user_interactions[user_interactions > 0].index.tolist()

        # Calculate recommendation scores
        recommendation_scores = {}
        for post in interacted_posts:
            # Get similarity scores for this post
            post_idx = self.user_post_matrix.columns.get_loc(post)
            similar_posts = self.similarity_matrix[post_idx]

            # Add scores for non-interacted posts
            for i, score in enumerate(similar_posts):
                post_id = self.user_post_matrix.columns[i]
                if post_id not in interacted_posts:
                    if post_id not in recommendation_scores:
                        recommendation_scores[post_id] = 0
                    recommendation_scores[post_id] += score

        # Sort recommendations by score in descending order
        recommended_posts = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [post_id for post_id, _ in recommended_posts]


# Example Usage
if __name__ == "__main__":
    recommender = CollaborativeFilteringRecommender("../data/processed/interaction_df.csv")
    user_id = 1  # Replace with a valid user ID
    recommendations = recommender.recommend_posts(user_id)
    print(f"Collaborative Filtering Recommendations for User {user_id}: {recommendations}")
