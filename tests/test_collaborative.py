from src.recommendation_engine.collaborative import CollaborativeRecommender

# Path to interactions CSV file
interactions_csv_path = "data/processed/interaction_df.csv"

# Initialize the Collaborative Recommender
collaborative_recommender = CollaborativeRecommender(interactions_csv_path)

# Test with a valid user ID
valid_user_id = 1  # Replace with a valid user ID
print(f"Testing Collaborative Recommender with valid user_id: {valid_user_id}")
valid_recommendations = collaborative_recommender.recommend(valid_user_id, top_n=10)
print(f"Collaborative Recommendations for user_id {valid_user_id}:\n{valid_recommendations}")

# Test with an invalid user ID
invalid_user_id = 9999  # Use an ID that doesn't exist in the dataset
print(f"\nTesting Collaborative Recommender with invalid user_id: {invalid_user_id}")
invalid_recommendations = collaborative_recommender.recommend(invalid_user_id, top_n=10)
print(f"Collaborative Recommendations for user_id {invalid_user_id}:\n{invalid_recommendations}")
