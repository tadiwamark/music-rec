"""
Group Members

Tungamiraishe Mukwena R204452G HAI
Nyasha Zhou R204449M HAI
Tadiwanashe Nyaruwata R204445V HAI
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cache data loading to prevent reloading on each run
@st.cache_data
def load_data():
    songs = pd.read_csv('songdata.csv')
    # Fixed random_state for consistent sampling
    songs = songs.sample(n=5000, random_state=42).drop('link', axis=1).reset_index(drop=True)
    songs['text'] = songs['text'].str.replace(r'\n', '')
    return songs

# Cache similarities computation to avoid recalculating
@st.cache_data
def compute_similarities(songs):
    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    lyrics_matrix = tfidf.fit_transform(songs['text'])
    cosine_similarities = cosine_similarity(lyrics_matrix)
    similarities = {}
    for i in range(len(cosine_similarities)):
        similar_indices = cosine_similarities[i].argsort()[:-50:-1]
        key = f"{songs['song'].iloc[i]} - {songs['artist'].iloc[i]}"
        similarities[key] = [(cosine_similarities[i][x], 
                            f"{songs['song'][x]} - {songs['artist'][x]}") 
                            for x in similar_indices][1:]
    return similarities

class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def recommend(self, song, number_songs):
        return self.matrix_similar.get(song, [])[:number_songs]

# Load cached data and compute similarities
songs = load_data()
similarities = compute_similarities(songs)
recommender = ContentBasedRecommender(similarities)

# Streamlit interface
st.title('Music Recommender System')
# Use a unique key for the selectbox to maintain state
selected_track = st.selectbox('Select a song', 
                             songs['song'] + " - " + songs['artist'], 
                             key='song_select')
number_songs = st.slider('Number of recommendations', 1, 10, 4, key='num_songs')

if st.button('Recommend'):
    results = recommender.recommend(selected_track, number_songs)
    if results:
        for idx, (score, track) in enumerate(results, 1):
            song, artist = track.split(" - ")
            st.write(f"Recommendation {idx}: {song} by {artist} (Score: {score:.3f})")
    else:
        st.error("No recommendations found. Try another song.")
