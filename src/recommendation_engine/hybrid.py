import pandas as pd

class HybridRecommender:
    def __init__(self, content_model, collaborative_model, weight_content=0.3, weight_collaborative=0.7):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.weight_content = weight_content
        self.weight_collaborative = weight_collaborative

    def recommend_hybrid(self, user_id, top_n=10):
        print(f"Generating hybrid recommendations for user: {user_id}")

        try:
            # Get recommendations from both models
            content_recommendations = self.content_model.recommend(user_id, top_n=top_n)
            collaborative_recommendations = self.collaborative_model.recommend(user_id, top_n=top_n)

            print(f"Content-Based Recommendations: {content_recommendations}")
            print(f"Collaborative Recommendations: {collaborative_recommendations}")

            # Check if content_recommendations is empty
            content_df = content_recommendations if not content_recommendations.empty else pd.DataFrame(columns=["post_id", "score"])

            # Handle collaborative recommendations (if list or DataFrame)
            if isinstance(collaborative_recommendations, list):
                collaborative_df = pd.DataFrame(
                    collaborative_recommendations,
                    columns=["post_id", "score"] if isinstance(collaborative_recommendations[0], tuple) else ["post_id"]
                )
                if "score" not in collaborative_df.columns:
                    collaborative_df["score"] = 1.0  # Assign default score
            else:
                collaborative_df = collaborative_recommendations if not collaborative_recommendations.empty else pd.DataFrame(columns=["post_id", "score"])

            # Add weights
            if not content_df.empty:
                content_df["weight"] = self.weight_content
                content_df["weighted_score"] = content_df["score"] * content_df["weight"]

            if not collaborative_df.empty:
                collaborative_df["weight"] = self.weight_collaborative
                collaborative_df["weighted_score"] = collaborative_df["score"] * collaborative_df["weight"]

            # Combine DataFrames safely
            combined_df = pd.concat([content_df, collaborative_df], ignore_index=True)

            if combined_df.empty:
                print("No recommendations available.")
                return pd.DataFrame(columns=["post_id", "weighted_score"])

            # Aggregate and sort recommendations
            final_recommendations = (
                combined_df.groupby("post_id", as_index=False)["weighted_score"]
                .sum()
                .sort_values(by="weighted_score", ascending=False)
                .head(top_n)
            )

            print(f"Final Hybrid Recommendations: {final_recommendations}")
            return final_recommendations

        except Exception as e:
            print(f"Error fetching recommendations: {e}")
            raise


