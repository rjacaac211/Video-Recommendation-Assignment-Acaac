from src.recommendation_engine.content_based import ContentBasedRecommender

# Path to your posts CSV file
posts_csv_path = "data/processed/all_posts_with_features.csv"

# Initialize the ContentBasedRecommender
content_recommender = ContentBasedRecommender(posts_csv_path)



# Choose a valid post_id from the dataset
valid_post_id = content_recommender.posts_df['id'].iloc[0]  # Use the first post ID in the dataset
print(f"Testing valid post_id: {valid_post_id}")

# Generate recommendations
valid_recommendations = content_recommender.recommend(valid_post_id, top_n=5)
print(f"Recommendations for valid post_id {valid_post_id}:\n{valid_recommendations}")



# Use a post_id that doesn't exist in the dataset
invalid_post_id = 9999  # Ensure this ID is not in your dataset
print(f"Testing invalid post_id: {invalid_post_id}")

# Generate recommendations
invalid_recommendations = content_recommender.recommend(invalid_post_id, top_n=5)
print(f"Recommendations for invalid post_id {invalid_post_id}:\n{invalid_recommendations}")



import pandas as pd

# Create an empty dataset and save to CSV
empty_posts_path = "empty_posts.csv"
pd.DataFrame(columns=["id", "title", "category_name"]).to_csv(empty_posts_path, index=False)

# Initialize recommender with an empty dataset
empty_recommender = ContentBasedRecommender(empty_posts_path)
print("Testing with empty dataset:")

# Test with a random post_id
empty_recommendations = empty_recommender.recommend(1, top_n=5)
print(f"Recommendations from empty dataset:\n{empty_recommendations}")



# Create a dataset with missing values
data_with_missing_values = pd.DataFrame({
    "id": [1, 2, 3],
    "title": ["Post A", None, "Post C"],  # Missing title for post_id 2
    "category_name": ["Category 1", "Category 2", None]  # Missing category for post_id 3
})
data_with_missing_values_path = "missing_values_posts.csv"
data_with_missing_values.to_csv(data_with_missing_values_path, index=False)

# Initialize recommender with missing values dataset
missing_values_recommender = ContentBasedRecommender(data_with_missing_values_path)
print("Testing with missing values dataset:")

# Test with a valid post_id
missing_values_recommendations = missing_values_recommender.recommend(1, top_n=5)
print(f"Recommendations with missing values dataset:\n{missing_values_recommendations}")



# Create a dataset with duplicate titles
data_with_duplicates = pd.DataFrame({
    "id": [1, 2, 3],
    "title": ["Duplicate Title", "Duplicate Title", "Unique Title"],
    "category_name": ["Category 1", "Category 2", "Category 3"]
})
data_with_duplicates_path = "duplicates_posts.csv"
data_with_duplicates.to_csv(data_with_duplicates_path, index=False)

# Initialize recommender with duplicate titles dataset
duplicates_recommender = ContentBasedRecommender(data_with_duplicates_path)
print("Testing with duplicate titles dataset:")

# Test with a valid post_id
duplicates_recommendations = duplicates_recommender.recommend(1, top_n=5)
print(f"Recommendations with duplicate titles dataset:\n{duplicates_recommendations}")



