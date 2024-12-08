# save_model.py
import joblib
import os
from recommendation_engine.content_based import ContentBasedRecommender
from recommendation_engine.collaborative import CollaborativeRecommender
from recommendation_engine.hybrid import HybridRecommender

# Define the path to the models directory
models_dir = "../models"

# Ensure the models directory exists, create it if not
os.makedirs(models_dir, exist_ok=True)

def save_content_model(content_model, filename='content_model.pkl'):
    try:
        # Save the content model in the models directory
        model_path = os.path.join(models_dir, filename)
        joblib.dump(content_model, model_path)
        print(f"Content model saved to {model_path}")
    except Exception as e:
        print(f"Error saving content model: {e}")

def save_collaborative_model(collaborative_model, filename='collaborative_model.pkl'):
    try:
        # Save the collaborative model in the models directory
        model_path = os.path.join(models_dir, filename)
        joblib.dump(collaborative_model, model_path)
        print(f"Collaborative model saved to {model_path}")
    except Exception as e:
        print(f"Error saving collaborative model: {e}")

def save_hybrid_model(hybrid_model, filename='hybrid_model.pkl'):
    try:
        # Save the hybrid model in the models directory
        model_path = os.path.join(models_dir, filename)
        joblib.dump(hybrid_model, model_path)
        print(f"Hybrid model saved to {model_path}")
    except Exception as e:
        print(f"Error saving hybrid model: {e}")

# Example of saving the models
if __name__ == "__main__":
    # Initialize your models here
    content_recommender = ContentBasedRecommender("../data/processed/all_posts_with_features.csv")
    collaborative_recommender = CollaborativeRecommender("../data/processed/interaction_df.csv")
    
    # Create hybrid recommender using the content and collaborative models
    hybrid_recommender = HybridRecommender(content_recommender, collaborative_recommender)
    
    # Save the models
    save_content_model(content_recommender, 'content_model.pkl')
    save_collaborative_model(collaborative_recommender, 'collaborative_model.pkl')
    save_hybrid_model(hybrid_recommender, 'hybrid_model.pkl')
