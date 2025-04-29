import streamlit as st
import pandas as pd
import numpy as np

# Placeholder function for the recommendation engine
def recommend_anime(user_inputs):
    # TODO: Replace this with your ML model / similarity matching logic
    # This is dummy data for demonstration purposes
    dummy_anime_data = {
        "Title": ["Attack on Titan", "Death Note", "Spirited Away"],
        "Genre": ["Action, Drama", "Mystery, Thriller", "Fantasy, Adventure"],
        "Why": ["Dark themes & worldbuilding", "Moral dilemmas & pacing", "Whimsical yet emotional"]
    }
    return pd.DataFrame(dummy_anime_data)

# App layout
st.set_page_config(page_title="Anime Recommender for Non-Anime Fans", layout="wide")

st.title("🎌 Anime Recommendations for Non-Anime Viewers")
st.markdown("""
Welcome! This tool recommends **anime shows or movies** based on the **non-anime content you already enjoy**.

This project explores the globalization of anime by drawing connections between anime and mainstream Western media.
""")

# Input form
with st.form("recommendation_form"):
    st.subheader("Tell us what you watch 🎥📺")
    user_shows = st.text_area("Enter a list of non-anime films or TV shows you enjoy (separated by commas)", 
                              placeholder="e.g. Breaking Bad, The Witcher, Black Mirror")

    submit = st.form_submit_button("Recommend me anime!")

# When the form is submitted
if submit and user_shows:
    st.subheader("🔍 Recommendations Based on Your Input")
    
    # Convert user input into list
    user_inputs = [s.strip() for s in user_shows.split(",") if s.strip()]
    
    # Generate recommendations (replace with your own function)
    recommendations = recommend_anime(user_inputs)
    
    # Display results
    st.dataframe(recommendations)
    st.markdown("✍️ *These recommendations are based on shared themes, tone, or genre similarities.*")

st.markdown("---")
st.markdown("👩‍💻 **About this project**")
st.markdown("""
This project was created as a final project for an anime course, exploring how anime's themes and genres align with global media trends. 

It uses data from **IMDb** and **MyAnimeList**, and is built with **Python** and **Streamlit**.
""")
