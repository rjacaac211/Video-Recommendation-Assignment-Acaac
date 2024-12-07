from flask import Flask, request, jsonify
from recommendation_engine.content_based import ContentBasedRecommender
from recommendation_engine.collaborative import CollaborativeFilteringRecommender
from recommendation_engine.hybrid import HybridRecommender
from recommendation_engine.cold_start import ColdStartRecommender

app = Flask(__name__)

# Load preprocessed data
CONTENT_DATA_PATH = "../data/processed/all_posts_with_features.csv"
INTERACTION_DATA_PATH = "../data/processed/interaction_df.csv"

# Initialize recommendation systems (Consider loading models into memory once)
content_recommender = ContentBasedRecommender(CONTENT_DATA_PATH)
collaborative_recommender = CollaborativeFilteringRecommender(INTERACTION_DATA_PATH)
hybrid_recommender = HybridRecommender(content_recommender, collaborative_recommender)
cold_start_recommender = ColdStartRecommender(CONTENT_DATA_PATH)

@app.route('/feed', methods=['GET'])
def get_recommendations():
    """
    API endpoint to provide recommendations based on user preferences and optional parameters.
    Query Params:
        - username: The user's username.
        - category_id: Optional category to filter recommendations.
        - mood: Optional mood to filter recommendations.
    Returns:
        JSON response with a list of recommended posts.
    """
    username = request.args.get('username')
    category_id = request.args.get('category_id', type=int)
    mood = request.args.get('mood')

    if not username:
        return jsonify({"error": "Missing required parameter: username"}), 400

    try:
        # Check if the user exists in the interaction data (for collaborative filtering)
        user_exists = username in collaborative_recommender.user_post_matrix.index

        # Generate recommendations
        if user_exists:
            # Filter by category or mood if provided
            recommendations = hybrid_recommender.recommend_posts(username, category_id, mood)
        else:
            # Handle cold-start recommendations
            recommendations = cold_start_recommender.recommend(username, category_id, mood)

        # Convert recommendations to JSON
        response = recommendations.to_dict(orient="records")
        return jsonify({"recommendations": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
