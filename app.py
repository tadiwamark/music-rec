"""
Group Members

Tungamiraishe Mukwena R204452G HAI
Nyasha Zhou R204449M HAI
Tadiwanashe Nyaruwata R204445V HAI
"""


import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
def load_data():
    songs = pd.read_csv('songdata.csv')
    songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)
    songs['text'] = songs['text'].str.replace(r'\n', '')
    return songs

# Compute the similarities
def compute_similarities(songs):
    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    lyrics_matrix = tfidf.fit_transform(songs['text'])
    cosine_similarities = cosine_similarity(lyrics_matrix)
    similarities = {}
    for i in range(len(cosine_similarities)):
        similar_indices = cosine_similarities[i].argsort()[:-50:-1]
        # Use song name and artist name as the key
        key = songs['song'].iloc[i] + " - " + songs['artist'].iloc[i]
        similarities[key] = [(cosine_similarities[i][x], songs['song'][x] + " - " + songs['artist'][x]) for x in similar_indices][1:]
    return similarities

# The recommendation class
class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def recommend(self, song, number_songs):
        if song not in self.matrix_similar:
            return []
        return self.matrix_similar[song][:number_songs]

# Load the data
songs = load_data()
# Compute the similarities
similarities = compute_similarities(songs)
# Create the recommender
recommender = ContentBasedRecommender(similarities)

# Streamlit app
st.title('Music Recommender System')
song = st.selectbox('Select a song', songs['song'] + " - " + songs['artist'])
number_songs = st.slider('Number of recommendations', 1, 10, 4)

if st.button('Recommend'):
    results = recommender.recommend(song, number_songs)
    if results:
        for idx, (score, song) in enumerate(results, 1):
            # Split the song and artist for display
            song, artist = song.split(" - ")
            st.write(f"Recommendation {idx}: {song} by {artist} with similarity score {score:.3f}")
    else:
        st.write("No recommendations found. Please select a different song.")
