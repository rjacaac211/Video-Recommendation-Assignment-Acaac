# Socialverse Recommendation System

This project implements a recommendation system for the Socialverse platform using a hybrid approach that combines **content-based** and **collaborative** filtering methods. It utilizes user interactions, post features, and mood-based filtering to generate personalized recommendations for users. 

## Table of Contents
- [Running the Project](#running-the-project)
  - [Install Dependencies](#1-install-dependencies)
  - [Start the Flask Application](#2-start-the-flask-application)
  - [Access the API](#3-access-the-api)
- [Usage](#usage)
- [Stages of Development](#stages-of-development)
- [Data Preprocessing](#1-data-preprocessing)
- [Algorithm Development](#2-algorithm-development)
  - [Content-Based Filtering](#content-based-filtering)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Hybrid Model](#hybrid-model)
  - [Model Justification](#model-justification)
- [Evaluation Metrics](#3-evaluation-metrics)
- [API Development](#4-api-development)
  - [Get Recommended Posts with Category and Mood Filters](#1-get-recommended-posts-with-category-and-mood-filters)

## Running the Project

To run the project locally, follow the steps below:

### 1. Install Dependencies

Ensure you have Python 3.x installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start the Flask Application

Run the application using the following command:
```bash
python3 app.py
```
This will start the Flask server on `http://127.0.0.1:5000`.

### 3. Access the API

Once the application is running, you can interact with the API by sending a request to the /feed endpoint with the desired parameters.

### Usage

#### 1. Get Recommended Posts with Category and Mood Filters

**Endpoint**: `/feed`  
**Method**: `GET`

##### Request Parameters:
- `username` (required): The ID of the user (integer).
- `category_id` (optional): The ID of the category the user wants to view.
- `mood` (optional): The mood the user is currently in (e.g., `happy`, `passion`, etc.).

##### URL Example:
```
http://127.0.0.1:5000/feed?username=211&category_id=1&mood=passion
```

## Stages of Development

This project was developed in multiple stages. Below is a breakdown of the key stages and the corresponding sections in this document:

1. **Data Preprocessing** – Involves cleaning and transforming raw data into a usable format.
2. **Algorithm Development** – Focuses on creating the core recommendation algorithms using content-based, collaborative, and hybrid approaches.
3. **Evaluation Metrics** – Describes the methods used to evaluate the performance of the recommendation system.
4. **API Development** – Details the development of the API that allows interaction with the recommendation system.
   
Each stage builds upon the previous one, culminating in the deployment of the recommendation system with a working API.

## 1. Data Preprocessing

- **Data Fetching:** The `data_fetcher.py` script retrieves video metadata and user interaction data from APIs and saves it as JSON files.
  
- **User Data Processing:** The `preprocess_users` function cleans and processes user data by filling missing values, converting date fields, and dropping irrelevant columns.

- **Post Data Processing:** The `preprocess_posts` function normalizes categories, extracts mood-related features, and removes unnecessary columns.

- **Interaction Data Processing:** The `preprocess_interactions` function consolidates views, likes, inspirations, and ratings into a single DataFrame and removes duplicates.

- **Feature Aggregation:** The `aggregate_interactions` function calculates aggregated features (e.g., total interactions, average ratings) for both users and posts.

- **Data Saving:** Processed data is saved as CSV files for further analysis and modeling.


## 2. Algorithm Development

### Content-Based Filtering
The content-based recommender recommends posts based on their similarity to those the user has previously interacted with. The system utilizes TF-IDF for title and mood-based feature extraction, combined with one-hot encoding for categories. The cosine similarity between the features determines post relevance.

**Key Functions:**
- `recommend(post_id, top_n=10, category_id=None, mood=None)`: Recommends posts similar to a given post, with optional category and mood filters.
- `extract_moods(emotions)`: Extracts mood-related keywords from post summaries to enrich content features.

### Collaborative Filtering
The collaborative recommender generates recommendations based on user interactions. It leverages item-item similarity, where posts are recommended based on how similar they are to those the user has already interacted with.

**Key Functions:**
- `recommend(user_id, top_n=10)`: Suggests posts for a user based on interactions of similar users.
  
### Hybrid Model
The hybrid recommender combines content-based and collaborative approaches. It blends recommendations from both models, weighted by the desired importance of each.

**Key Functions:**
- `recommend_hybrid(user_id, category_id=None, top_n=10)`: Integrates content and collaborative recommendations, with weighting for each model.

### Model Justification
The hybrid approach improves recommendation accuracy by combining the strengths of both content-based and collaborative filtering, ensuring robust recommendations even for users with limited interaction history (cold start problem).


## 3. Evaluation Metrics

- **Mean Absolute Error (MAE):**
  - Measures the average magnitude of errors in predictions.
  - Formula: `MAE = (1/n) * Σ|y_true - y_pred|`
  
- **Root Mean Square Error (RMSE):**
  - Measures the square root of the average squared differences between predicted and actual ratings.
  - Formula: `RMSE = sqrt((1/n) * Σ(y_true - y_pred)^2)`
  
### Summary:
- **MAE** and **RMSE** are used to assess the accuracy of the hybrid recommendation model.
- The model's MAE is **21.26**, and its RMSE is **28.30**, indicating the average deviation from actual ratings.


## 4. API Development

### 1. Get Recommended Posts with Category and Mood Filters

**Endpoint**: `/feed`  
**Method**: `GET`

#### Request Parameters:
- `username` (required): The ID of the user (integer).
- `category_id` (optional): The ID of the category the user wants to view.
- `mood` (optional): The mood the user is currently in (e.g., `happy`, `passion`, etc.).

#### URL Example:
```
http://127.0.0.1:5000/feed?username=211&category_id=1&mood=passion
```


#### Response:
Returns the top 10 recommended posts for the user, filtered by the `category_id` and `mood` parameters.

**Response Format**:
```json
{
  "recommendations": [
    {
      "category_id": 1,
      "post_id": 366,
      "weighted_score": 0.45640826146457697
    },
    {
      "category_id": 1,
      "post_id": 17,
      "weighted_score": 0.27980624030397405
    },
    ...
  ]
}
