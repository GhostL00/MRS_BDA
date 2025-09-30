# 🎬 Movie Recommendation System (Content / Genre / Rating + Dashboards)
# Tech Stack: Python + Pandas + Streamlit + Scikit-learn + Matplotlib/Seaborn

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Load Datasets
# -------------------------------
movies = pd.read_csv("movies.csv")   # movieId, title, genres, description
ratings = pd.read_csv("ratings.csv") # userId, movieId, rating

# -------------------------------
# 2. Content-Based Filtering (Genres + Description)
# -------------------------------
movies["genres"] = movies["genres"].fillna("")
if "description" in movies.columns:
    movies["description"] = movies["description"].fillna("")
    movies["content"] = movies["title"] + " " + movies["genres"] + " " + movies["description"]
else:
    movies["content"] = movies["title"] + " " + movies["genres"]

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["content"])
cosine_sim_content = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def content_recommendations(title, num=5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_content[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]["title"].tolist()

# -------------------------------
# 3. Genre-Based Recommendations
# -------------------------------
def genre_recommendations(title, num=5):
    if title not in movies["title"].values:
        return []
    movie_genres = movies[movies["title"] == title]["genres"].values[0]
    genre_list = movie_genres.split("|")
    filtered = movies[movies["genres"].str.contains("|".join(genre_list))]
    top_recs = filtered.sample(min(num, len(filtered)))["title"].tolist()
    return top_recs

# -------------------------------
# 4. Rating-Based Recommendations
# -------------------------------
def rating_recommendations(num=5):
    avg_ratings = ratings.groupby("movieId")["rating"].mean()
    top_movies = avg_ratings.sort_values(ascending=False).head(num).index
    return movies[movies["movieId"].isin(top_movies)]["title"].tolist()

# -------------------------------
# 5. Streamlit App
# -------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎥 Movie Recommendation System")
st.write("Choose a recommendation mode: **Content-Based | Genre | Rating**")

# User Input
col1, col2 = st.columns(2)
with col1:
    movie_list = movies["title"].values
    selected_movie = st.selectbox("Choose a movie:", movie_list)

with col2:
    mode = st.radio("Recommendation Mode:", ["Content-Based", "Genre", "Rating"])

if st.button("Recommend"):
    if mode == "Content-Based":
        recommendations = content_recommendations(selected_movie, num=5)
    elif mode == "Genre":
        recommendations = genre_recommendations(selected_movie, num=5)
    else:
        recommendations = rating_recommendations(num=5)

    st.subheader(f"✅ {mode} Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

# -------------------------------
# 6. Visualization Dashboards
# -------------------------------
st.header("📊 Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["⭐ Rating Distribution", "🎭 Popular Genres", "🎬 Top Movies"])

# Rating Distribution
with tab1:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(ratings["rating"], bins=10, kde=False, ax=ax)
    ax.set_title("Rating Distribution")
    st.pyplot(fig)

# Popular Genres
with tab2:
    genre_counts = movies["genres"].str.split("|").explode().value_counts()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=genre_counts.values[:10], y=genre_counts.index[:10], ax=ax)
    ax.set_title("Top 10 Popular Genres")
    st.pyplot(fig)

# Top Movies by Average Rating
with tab3:
    top_movies = ratings.groupby("movieId")["rating"].mean().sort_values(ascending=False).head(10)
    top_movies = movies[movies["movieId"].isin(top_movies.index)][["title"]].assign(rating=top_movies.values)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=top_movies["rating"], y=top_movies["title"], ax=ax)
    ax.set_title("Top 10 Movies by Average Rating")
    st.pyplot(fig)

