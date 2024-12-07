import pandas as pd


class ColdStartRecommender:
    def __init__(self, posts_data_path):
        """
        Initialize the Cold Start Recommender.
        Args:
            posts_data_path (str): Path to the CSV file containing post data.
        """
        self.posts_df = pd.read_csv(posts_data_path)

    def recommend_by_category(self, category_id, top_n=10):
        """
        Recommend posts based on the specified category.
        Args:
            category_id (int): The ID of the category to recommend posts for.
            top_n (int): Number of posts to recommend.
        Returns:
            pd.DataFrame: A DataFrame of recommended posts.
        """
        filtered_posts = self.posts_df[self.posts_df['category_id'] == category_id]
        recommended_posts = filtered_posts.sort_values(
            by=['total_views', 'average_rating_features'],
            ascending=False
        ).head(top_n)
        return recommended_posts[['id', 'title', 'category_name', 'total_views', 'average_rating_features']]

    def recommend_by_popularity(self, top_n=10):
        """
        Recommend the most popular posts.
        Args:
            top_n (int): Number of posts to recommend.
        Returns:
            pd.DataFrame: A DataFrame of recommended posts.
        """
        recommended_posts = self.posts_df.sort_values(
            by=['total_views', 'total_likes', 'average_rating_features'],
            ascending=False
        ).head(top_n)
        return recommended_posts[['id', 'title', 'category_name', 'total_views', 'total_likes', 'average_rating_features']]

    def recommend_by_mood(self, mood, top_n=10):
        """
        Recommend posts based on the user's current mood.
        Args:
            mood (str): The mood of the user (e.g., 'happy', 'motivated').
            top_n (int): Number of posts to recommend.
        Returns:
            pd.DataFrame: A DataFrame of recommended posts.
        """
        mood_column = 'mood_tags'  # Assumes posts have mood-related tags
        filtered_posts = self.posts_df[self.posts_df[mood_column].str.contains(mood, na=False, case=False)]
        recommended_posts = filtered_posts.sort_values(
            by=['total_views', 'average_rating_features'],
            ascending=False
        ).head(top_n)
        return recommended_posts[['id', 'title', 'category_name', 'total_views', 'average_rating_features']]


# Example Usage
if __name__ == "__main__":
    cold_start = ColdStartRecommender("../data/processed/all_posts_with_features.csv")

    # Recommendations by category
    category_id = 1  # Replace with a valid category ID
    category_recommendations = cold_start.recommend_by_category(category_id, top_n=10)
    print("\nRecommendations by Category:")
    print(category_recommendations)

    # Recommendations by popularity
    popularity_recommendations = cold_start.recommend_by_popularity(top_n=10)
    print("\nRecommendations by Popularity:")
    print(popularity_recommendations)

    # Recommendations by mood
    user_mood = "motivated"  # Replace with a user mood
    mood_recommendations = cold_start.recommend_by_mood(user_mood, top_n=10)
    print("\nRecommendations by Mood:")
    print(mood_recommendations)
