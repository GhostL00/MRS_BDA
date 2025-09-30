# 🎬 Movie Recommendation System (Content | Genre by Movie | Genre by Choice | Rating + Dashboards)

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
# 2. Content-Based Filtering
# -------------------------------
movies["genres"] = movies["genres"].fillna("")
if "description" in movies.columns:
    movies["description"] = movies["description"].fillna("")
    movies["content"] = movies["title"] + " " + movies["genres"] + " " + movies["description"]
else:
    movies["content"] = movies["title"] + " " + movies["genres"]

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
# 3. Genre-Based (Movie’s Genres)
# -------------------------------
def genre_recommendations(title, num=5):
    if title not in movies["title"].values:
        return []
    movie_genres = movies[movies["title"] == title]["genres"].values[0]
    genre_list = movie_genres.split("|")
    filtered = movies[movies["genres"].str.contains("|".join(genre_list))]
    return filtered.sample(min(num, len(filtered)))["title"].tolist()

# -------------------------------
# 4. Genre-Based (User Choice)
# -------------------------------
def genre_choice_recommendations(genre, num=5):
    filtered = movies[movies["genres"].str.contains(genre, case=False, na=False)]
    avg_ratings = ratings.groupby("movieId")["rating"].mean()
    top_movies = avg_ratings[avg_ratings.index.isin(filtered["movieId"])].sort_values(ascending=False).head(num).index
    return movies[movies["movieId"].isin(top_movies)]["title"].tolist()

# -------------------------------
# 5. Rating-Based
# -------------------------------
def rating_recommendations(num=5):
    avg_ratings = ratings.groupby("movieId")["rating"].mean()
    top_movies = avg_ratings.sort_values(ascending=False).head(num).index
    return movies[movies["movieId"].isin(top_movies)]["title"].tolist()

# -------------------------------
# 6. Streamlit App
# -------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎥 Movie Recommendation System")
st.write("Choose a recommendation mode: **Content-Based | Genre by Movie | Genre by Choice | Rating**")

col1, col2 = st.columns(2)
with col1:
    movie_list = movies["title"].values
    selected_movie = st.selectbox("Choose a movie:", movie_list)

with col2:
    mode = st.radio("Recommendation Mode:", ["Content-Based", "Genre (Movie-Based)", "Genre (User Choice)", "Rating-Based"])

# Extra option if user chooses Genre (User Choice)
genre_selected = None
if mode == "Genre (User Choice)":
    all_genres = set(g for gs in movies["genres"].dropna().str.split("|") for g in gs)
    genre_selected = st.selectbox("Pick a genre:", sorted(all_genres))

if st.button("Recommend"):
    if mode == "Content-Based":
        recommendations = content_recommendations(selected_movie, num=5)
    elif mode == "Genre (Movie-Based)":
        recommendations = genre_recommendations(selected_movie, num=5)
    elif mode == "Genre (User Choice)" and genre_selected:
        recommendations = genre_choice_recommendations(genre_selected, num=5)
    else:
        recommendations = rating_recommendations(num=5)

    st.subheader(f"✅ {mode} Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

# -------------------------------
# 7. Visualization Dashboards
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
