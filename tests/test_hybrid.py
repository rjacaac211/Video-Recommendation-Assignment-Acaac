from src.recommendation_engine.content_based import ContentBasedRecommender
from src.recommendation_engine.collaborative import CollaborativeRecommender
from src.recommendation_engine.hybrid import HybridRecommender
import pandas as pd

# Paths to test datasets
posts_csv_path = "data/processed/all_posts_with_features.csv"
interactions_csv_path = "data/processed/interaction_df.csv"

# Initialize the Content-Based Recommender
content_recommender = ContentBasedRecommender(posts_csv_path)

# Check for a valid post_id in the content-based data
valid_content_post_id = None
if not content_recommender.posts_df.empty:
    valid_content_post_id = content_recommender.posts_df['id'].iloc[0]
    print(f"Valid Post ID Found for Content-Based Recommender: {valid_content_post_id}")
else:
    print("Content-Based Data is empty. Cannot find valid Post ID.")

# Initialize the Collaborative Recommender
collaborative_recommender = CollaborativeRecommender(interactions_csv_path)

# Initialize the Hybrid Recommender
hybrid_recommender = HybridRecommender(content_recommender, collaborative_recommender)

# Test the Hybrid Recommender with a valid user_id and valid post_id for content-based
print("\nTesting Hybrid Recommender with a valid user_id and content-based valid post_id:")
if valid_content_post_id:
    try:
        hybrid_recommendations = hybrid_recommender.recommend_hybrid(valid_content_post_id, top_n=5)
        print(f"Hybrid Recommendations for content-based valid post_id {valid_content_post_id}:\n{hybrid_recommendations}")
    except Exception as e:
        print(f"Error generating hybrid recommendations for valid content-based post_id: {e}")
else:
    print("Skipping this test due to missing valid Post ID in Content-Based Data.")

# Test with a valid user ID (from collaborative filtering data)
valid_user_id = 1  # Replace with a valid user ID from your dataset
print(f"\nTesting Hybrid Recommender with valid user_id: {valid_user_id}")
try:
    hybrid_recommendations = hybrid_recommender.recommend_hybrid(valid_user_id, top_n=5)
    print(f"Hybrid Recommendations for user_id {valid_user_id}:\n{hybrid_recommendations}")
except Exception as e:
    print(f"Error generating hybrid recommendations: {e}")

# Test with an invalid user ID
invalid_user_id = 9999  # Use an ID that doesn't exist in the interactions dataset
print(f"\nTesting Hybrid Recommender with invalid user_id: {invalid_user_id}")
try:
    hybrid_recommendations = hybrid_recommender.recommend_hybrid(invalid_user_id, top_n=5)
    print(f"Hybrid Recommendations for user_id {invalid_user_id}:\n{hybrid_recommendations}")
except Exception as e:
    print(f"Error generating hybrid recommendations: {e}")

# Test with edge case (empty dataset)
print("\nTesting Hybrid Recommender with empty dataset:")
empty_content_csv_path = "empty_posts.csv"
pd.DataFrame(columns=["id", "title", "category_name"]).to_csv(empty_content_csv_path, index=False)

empty_content_recommender = ContentBasedRecommender(empty_content_csv_path)
hybrid_with_empty_content = HybridRecommender(empty_content_recommender, collaborative_recommender)

try:
    hybrid_recommendations_empty = hybrid_with_empty_content.recommend_hybrid(valid_user_id, top_n=5)
    print(f"Hybrid Recommendations with empty dataset:\n{hybrid_recommendations_empty}")
except Exception as e:
    print(f"Error generating hybrid recommendations with empty dataset: {e}")
