# 🎬 Movie Recommendation System - Mini Project
# Tech Stack: Python + Streamlit + Scikit-learn
# Author: Your Name

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Load Dataset
# -------------------------------
# You can download MovieLens dataset or make your own CSV
# CSV should have: movieId, title, genres, description(optional)
movies = pd.read_csv("movies.csv")  

# Example columns: [movieId, title, genres, description]
# If no description column, we will only use title+genres

# -------------------------------
# 2. Preprocessing
# -------------------------------
# Fill missing values
movies["genres"] = movies["genres"].fillna("")
if "description" in movies.columns:
    movies["description"] = movies["description"].fillna("")
    movies["content"] = movies["title"] + " " + movies["genres"] + " " + movies["description"]
else:
    movies["content"] = movies["title"] + " " + movies["genres"]

# -------------------------------
# 3. Feature Extraction (TF-IDF)
# -------------------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["content"])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Helper: Get movie index
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def recommend_movie(title, num_recommendations=5):
    if title not in indices:
        return ["Movie not found in dataset!"]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # skip the movie itself
    movie_indices = [i[0] for i in sim_scores]
    
    return movies["title"].iloc[movie_indices].tolist()

# -------------------------------
# 4. Streamlit Web App
# -------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎥 Movie Recommendation System")
st.write("Get movie suggestions based on your favorite movie!")

# Dropdown to select movie
movie_list = movies["title"].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    recommendations = recommend_movie(selected_movie, 5)
    st.subheader("✅ Recommended Movies:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
