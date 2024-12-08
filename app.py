import os
import sys
from flask import Flask, request, jsonify
from src.recommendation_engine.content_based import ContentBasedRecommender
from src.recommendation_engine.collaborative import CollaborativeRecommender
from src.recommendation_engine.hybrid import HybridRecommender

# Add src directory to the Python path (if needed)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Initialize Flask app
app = Flask(__name__)

# Load preprocessed data paths
CONTENT_DATA_PATH = "data/processed/all_posts_with_features.csv"
INTERACTION_DATA_PATH = "data/processed/interaction_df.csv"

# Initialize recommendation systems
content_recommender = ContentBasedRecommender(CONTENT_DATA_PATH)
collaborative_recommender = CollaborativeRecommender(INTERACTION_DATA_PATH)

# Initialize the Hybrid Recommender
hybrid_recommender = HybridRecommender(
    content_model=content_recommender,
    collaborative_model=collaborative_recommender
)

@app.route('/feed', methods=['GET'])
def get_recommendations():
    username = request.args.get('username')
    category_id = request.args.get('category_id', type=int)
    mood = request.args.get('mood')

    if not username:
        return jsonify({"error": "Missing required parameter: username"}), 400

    try:
        # Convert username to integer (if possible)
        try:
            username = int(username)
        except ValueError:
            return jsonify({"error": "Invalid username format. It must be an integer."}), 400

        print(f"API Username Provided: {username} (Type: {type(username)})")

        # Validate if user exists in the interaction matrix
        if collaborative_recommender.user_post_matrix.empty:
            print("Error: User-Post Interaction Matrix is empty.")
            return jsonify({"error": "User-Post Interaction Matrix is not loaded properly"}), 500

        if username not in collaborative_recommender.user_post_matrix.index:
            print(f"Error: Username {username} not found in the interaction matrix.")
            return jsonify({"error": f"No data found for user {username}"}), 404

        # Generate recommendations based on filters (category_id, mood)
        if category_id:
            recommendations = hybrid_recommender.recommend_hybrid(username, top_n=10)
            recommendations = recommendations[recommendations['category_id'] == category_id]
        elif mood:
            recommendations = hybrid_recommender.recommend_hybrid(username, top_n=10)
            recommendations = recommendations[recommendations['mood_tags'].str.contains(mood, na=False, case=False)]
        else:
            recommendations = hybrid_recommender.recommend_hybrid(username, top_n=10)

        # Check for empty recommendations
        if recommendations.empty or recommendations is None:
            print(f"No recommendations available for user {username} with provided filters.")
            return jsonify({"error": "No recommendations available"}), 404

        # Convert recommendations to JSON
        response = recommendations.to_dict(orient="records")
        return jsonify({"recommendations": response})

    except Exception as e:
        print(f"Error encountered: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
