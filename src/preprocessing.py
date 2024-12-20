import os
import json
import pandas as pd

# Define paths
DATA_DIR = "../data"
PROCESSED_DIR = "../data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)  # Ensure the processed directory exists

# Helper Functions
def load_json_to_df(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_users(users_df):
    users_df.fillna({
        "instagram-url": "Not Provided",
        "youtube_url": "Not Provided",
        "tictok_url": "Not Provided"
    }, inplace=True)
    users_df['last_login'] = pd.to_datetime(users_df['last_login'], errors='coerce').fillna("Never Logged In")
    drop_cols = [
        'first_name', 'last_name', 'profile_url', 'bio',
        'website_url', 'instagram-url', 'youtube_url',
        'tictok_url', 'latitude', 'longitude'
    ]
    users_df.drop(columns=drop_cols, inplace=True)
    return users_df

def extract_moods(emotions):
    moods = []

    # If emotions is a dictionary, process its keys that contain lists, strings, or boolean values
    if isinstance(emotions, dict):
        for key, value in emotions.items():
            if isinstance(value, list):  # If the value is a list (e.g., moods)
                for item in value:
                    moods.append(str(item) if isinstance(item, (str, int, float)) else str(item))
            elif isinstance(value, dict):  # If the value is another dictionary, recursively extract moods
                moods.extend(extract_moods(value))
            elif isinstance(value, str):  # If the value is a string (e.g., initial, mid, final emotions)
                moods.append(str(value))
            elif isinstance(value, bool) and value:  # If the value is a boolean (True), add the key as a mood
                moods.append(key)
    elif isinstance(emotions, list):  # If emotions is a list, process it directly
        for item in emotions:
            moods.append(str(item) if isinstance(item, (str, int, float)) else str(item))

    return ', '.join(moods) if moods else None



def preprocess_posts(posts_df):
    posts_df['created_at'] = pd.to_datetime(posts_df['created_at'], errors='coerce')

    # Extract category data
    category_df = pd.json_normalize(posts_df['category'])
    posts_df = pd.concat([posts_df.drop(columns=['category']), category_df.add_prefix('category_')], axis=1)

    # Extract moods from the 'emotions' field inside 'post_summary'
    if 'post_summary' in posts_df.columns:
        # Apply the extract_moods function to 'emotions' in 'post_summary'
        posts_df['moods'] = posts_df['post_summary'].apply(
            lambda x: extract_moods(x.get('emotions', {})) if isinstance(x, dict) and 'emotions' in x else None
        )
    else:
        posts_df['moods'] = None  # If 'post_summary' doesn't exist, set 'moods' to None

    # Fill missing moods with 'Unknown'
    posts_df['moods'].fillna("Unknown", inplace=True)

    # Drop unnecessary columns
    drop_cols = [
        'slug', 'identifier', 'comment_count', 'exit_count',
        'thumbnail_url', 'gif_thumbnail_url', 'picture_url',
        'post_summary', 'category_count', 'category_description', 'category_image_url'
    ]
    posts_df.drop(columns=drop_cols, inplace=True)

    return posts_df




def preprocess_interactions(viewed, liked, inspired, rated):
    for df, interaction in zip(
        [viewed, liked, inspired, rated],
        ['viewed', 'liked', 'inspired', 'rated']
    ):
        df['interaction_type'] = interaction
        if interaction != 'rated':
            df['rating_percent'] = None
        df[f"{interaction}_at"] = pd.to_datetime(df[f"{interaction}_at"], errors='coerce')
        df.drop_duplicates(inplace=True)

    interaction_df = pd.concat([viewed, liked, inspired, rated], ignore_index=True)
    interaction_df.sort_values(by=['user_id', 'viewed_at', 'liked_at', 'inspired_at', 'rated_at'], inplace=True)
    interaction_df.reset_index(drop=True, inplace=True)
    return interaction_df

def aggregate_interactions(interaction_df):
    user_features = interaction_df.groupby('user_id').agg(
        total_views=('interaction_type', lambda x: (x == 'viewed').sum()),
        total_likes=('interaction_type', lambda x: (x == 'liked').sum()),
        total_inspirations=('interaction_type', lambda x: (x == 'inspired').sum()),
        total_ratings=('interaction_type', lambda x: (x == 'rated').sum()),
        average_rating=('rating_percent', 'mean')
    ).fillna(0).reset_index()

    post_features = interaction_df.groupby('post_id').agg(
        total_views=('interaction_type', lambda x: (x == 'viewed').sum()),
        total_likes=('interaction_type', lambda x: (x == 'liked').sum()),
        total_inspirations=('interaction_type', lambda x: (x == 'inspired').sum()),
        total_ratings=('interaction_type', lambda x: (x == 'rated').sum()),
        average_rating=('rating_percent', 'mean')
    ).fillna(0).reset_index()

    return user_features, post_features

def save_to_csv(df, filename):
    filepath = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filename} to {PROCESSED_DIR}")

# Main Preprocessing Pipeline
def main():
    print("Starting preprocessing pipeline...")
    viewed_df = load_json_to_df("viewed_posts.json")
    liked_df = load_json_to_df("liked_posts.json")
    inspired_df = load_json_to_df("inspired_posts.json")
    rated_df = load_json_to_df("rated_posts.json")
    all_posts_df = load_json_to_df("all_posts.json")
    all_users_df = load_json_to_df("all_users.json")

    print("Data loading complete.")

    all_users_df = preprocess_users(all_users_df)
    all_posts_df = preprocess_posts(all_posts_df)
    interaction_df = preprocess_interactions(viewed_df, liked_df, inspired_df, rated_df)
    user_features, post_features = aggregate_interactions(interaction_df)

    all_posts_with_features = pd.merge(
        all_posts_df,
        post_features.rename(columns={"average_rating": "average_rating_features"}),
        left_on='id',
        right_on='post_id',
        how='left'
    ).fillna(0)

    save_to_csv(interaction_df, "interaction_df.csv")
    save_to_csv(all_posts_with_features, "all_posts_with_features.csv")
    save_to_csv(all_users_df, "all_users_processed.csv")

    print("Preprocessing pipeline completed successfully!")

if __name__ == "__main__":
    main()
