from src.recommendation_engine.content_based import ContentBasedRecommender
import pandas as pd

# Path to your posts CSV file
posts_csv_path = "data/processed/all_posts_with_features.csv"

# Initialize the ContentBasedRecommender
content_recommender = ContentBasedRecommender(posts_csv_path)

# 1. Test with a valid post_id from the dataset
valid_post_id = content_recommender.posts_df['id'].iloc[0]  # Use the first post ID in the dataset
print(f"Testing valid post_id: {valid_post_id}")

# Generate recommendations
valid_recommendations = content_recommender.recommend(valid_post_id, top_n=10)
print(f"Recommendations for valid post_id {valid_post_id}:\n{valid_recommendations}")

# 2. Test with an invalid post_id (not present in the dataset)
invalid_post_id = 9999  # Ensure this ID is not in your dataset
print(f"Testing invalid post_id: {invalid_post_id}")

# Generate recommendations
invalid_recommendations = content_recommender.recommend(invalid_post_id, top_n=10)
print(f"Recommendations for invalid post_id {invalid_post_id}:\n{invalid_recommendations}")

# 3. Test with an empty dataset
empty_posts_path = "tests/outputs/empty_posts.csv"
pd.DataFrame(columns=["id", "title", "category_name", "moods"]).to_csv(empty_posts_path, index=False)  # Include moods column

empty_recommender = ContentBasedRecommender(empty_posts_path)
print("Testing with empty dataset:")

# Generate recommendations
empty_recommendations = empty_recommender.recommend(1, top_n=10)
print(f"Recommendations from empty dataset:\n{empty_recommendations}")

# 4. Test with a dataset containing missing values
data_with_missing_values = pd.DataFrame({
    "id": [1, 2, 3],
    "title": ["Post A", None, "Post C"],  # Missing title for post_id 2
    "category_name": ["Category 1", "Category 2", None],  # Missing category for post_id 3
    "moods": ["happy, excited", "sad, contemplative", None]  # Missing moods for post_id 3
})
data_with_missing_values_path = "tests/outputs/missing_values_posts.csv"
data_with_missing_values.to_csv(data_with_missing_values_path, index=False)

missing_values_recommender = ContentBasedRecommender(data_with_missing_values_path)
print("Testing with missing values dataset:")

# Test with a valid post_id
missing_values_recommendations = missing_values_recommender.recommend(1, top_n=10)
print(f"Recommendations with missing values dataset:\n{missing_values_recommendations}")

# 5. Test with a dataset containing duplicate titles
data_with_duplicates = pd.DataFrame({
    "id": [1, 2, 3],
    "title": ["Duplicate Title", "Duplicate Title", "Unique Title"],
    "category_name": ["Category 1", "Category 2", "Category 3"],
    "moods": ["energetic, motivated", "calm, reflective", "unique"]
})
data_with_duplicates_path = "tests/outputs/duplicates_posts.csv"
data_with_duplicates.to_csv(data_with_duplicates_path, index=False)

duplicates_recommender = ContentBasedRecommender(data_with_duplicates_path)
print("Testing with duplicate titles dataset:")

# Test with a valid post_id
duplicates_recommendations = duplicates_recommender.recommend(1, top_n=10)
print(f"Recommendations with duplicate titles dataset:\n{duplicates_recommendations}")

# 6. Test with moods affecting similarity
data_with_moods = pd.DataFrame({
    "id": [1, 2, 3],
    "title": ["Energetic Start", "Calm Evening", "Motivational Speech"],
    "category_name": ["Category A", "Category B", "Category C"],
    "moods": ["energetic, motivated", "calm, peaceful", "motivational, inspiring"]
})
data_with_moods_path = "tests/outputs/moods_posts.csv"
data_with_moods.to_csv(data_with_moods_path, index=False)

moods_recommender = ContentBasedRecommender(data_with_moods_path)
print("Testing moods integration:")

# Test with a valid post_id
moods_recommendations = moods_recommender.recommend(1, top_n=10)
print(f"Recommendations based on moods:\n{moods_recommendations}")

# Test with a new user who has no interaction history
new_user_mood = "passion"  # Assuming the new user selects this mood
recommendations = content_recommender._recommend_cold_start(user_mood=new_user_mood, top_n=10)
print(f"Cold Start Recommendations for mood '{new_user_mood}':\n{recommendations}")

