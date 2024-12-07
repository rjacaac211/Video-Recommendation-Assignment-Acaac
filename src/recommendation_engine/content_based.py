import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

class ContentBasedRecommender:
    def __init__(self, posts_path):
        self.posts_df = pd.read_csv(posts_path)
        self._prepare_data()

    def _prepare_data(self):
        if self.posts_df.empty:
            print("Posts DataFrame is empty. Skipping data preparation.")
            self.title_tfidf_matrix = None
            self.category_encoded = None
            self.combined_features = None
            self.similarity_matrix = None
            return

        # Fill missing titles
        self.posts_df['title'] = self.posts_df['title'].fillna("Unknown")

        # Preprocess text
        self.posts_df['processed_title'] = self.posts_df['title'].apply(self._preprocess_text)

        # Check if all processed titles are empty (e.g., due to stop words)
        if self.posts_df['processed_title'].str.strip().eq("").all():
            print("Processed titles are empty. Skipping data preparation.")
            self.title_tfidf_matrix = None
            self.category_encoded = None
            self.combined_features = None
            self.similarity_matrix = None
            return

        # Vectorize titles
        self.tfidf = TfidfVectorizer()
        self.title_tfidf_matrix = self.tfidf.fit_transform(self.posts_df['processed_title'])

        # One-hot encode categories
        self.category_encoded = pd.get_dummies(self.posts_df['category_name'])

        # Combine features
        self.combined_features = hstack([self.title_tfidf_matrix, self.category_encoded.values])

        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.combined_features, self.combined_features)


    def _preprocess_text(self, text):
        import string
        from nltk.corpus import stopwords
        from nltk.tokenize import TreebankWordTokenizer
        from nltk.stem import WordNetLemmatizer
        import nltk

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        tokenizer = TreebankWordTokenizer()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        # Text cleaning and tokenization
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = [word for word in tokenizer.tokenize(text) if word not in stop_words]
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(lemmatized)

    def recommend(self, post_id, top_n=10):
        if self.similarity_matrix is None:
            print("Similarity matrix not computed. Unable to provide recommendations.")
            return pd.DataFrame()

        if post_id not in self.posts_df['id'].values:
            print(f"Post ID {post_id} not found in posts data.")
            return pd.DataFrame(columns=['post_id', 'score'])

        print(f"Generating recommendations for post_id: {post_id}")
        try:
            # Locate the post index
            post_index = self.posts_df[self.posts_df['id'] == post_id].index[0]
            print(f"Post Index: {post_index}")

            # Calculate similarity scores
            similarity_scores = list(enumerate(self.similarity_matrix[post_index]))
            print(f"Similarity Scores (Raw): {similarity_scores[:5]}")

            # Sort and retrieve top N recommendations
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
            recommendations = [self.posts_df.iloc[i[0]]['id'] for i in similarity_scores]
            print(f"Top-{top_n} Recommendations: {recommendations}")

            return pd.DataFrame({'post_id': recommendations, 'score': [score for _, score in similarity_scores]})

        except Exception as e:
            print(f"Error in Content-Based Recommendation: {e}")
            return pd.DataFrame()


