import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, posts_data_path):
        """
        Initialize the Content-Based Recommender.
        Args:
            posts_data_path (str): Path to the CSV file containing post metadata.
        """
        self.posts_df = pd.read_csv(posts_data_path)
        self.similarity_matrix = None

    def preprocess_titles(self):
        """
        Preprocess the post titles for TF-IDF vectorization.
        """
        # Handle missing titles and convert to lowercase
        self.posts_df['title'] = self.posts_df['title'].fillna("Unknown").str.lower()

    def compute_similarity(self):
        """
        Compute the similarity matrix using TF-IDF for post titles.
        """
        self.preprocess_titles()
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.posts_df['title'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

    def get_similar_posts(self, post_id, top_n=10):
        """
        Get top-N similar posts for a given post ID based on content.
        Args:
            post_id (int): The ID of the post to find similar posts for.
            top_n (int): Number of similar posts to return.
        Returns:
            list: List of recommended post IDs.
        """
        if self.similarity_matrix is None:
            self.compute_similarity()

        # Find the index of the given post ID
        try:
            post_idx = self.posts_df[self.posts_df['id'] == post_id].index[0]
        except IndexError:
            raise ValueError(f"Post ID {post_id} not found in the dataset.")

        # Retrieve similarity scores
        similarity_scores = list(enumerate(self.similarity_matrix[post_idx]))

        # Sort by similarity score in descending order and exclude itself
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        # Retrieve post IDs of the most similar posts
        similar_post_ids = [self.posts_df.iloc[i[0]]['id'] for i in similarity_scores]
        return similar_post_ids
