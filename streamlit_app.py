import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

tv_df = pd.read_csv("data/non_anime.csv")
anime_df = pd.read_csv("data/anime_metadata.csv")

def score_to_color(score):
    norm_score = max(0, min(score / 100, 1))

    red = int(255 * (1 - norm_score))
    green = int(255 * norm_score)
    color = f'rgb({red},{green},0)'

    return color

def recommend_anime(user_inputs, k=5):
    user_show_descriptions = tv_df[tv_df['title'].isin(user_inputs)]['overview'].tolist()

    anime_embeddings = model.encode(anime_df['description'].tolist(), convert_to_tensor=True)

    user_embeddings = model.encode(user_show_descriptions, convert_to_tensor=True)

    user_embedding = torch.mean(user_embeddings, dim=0).unsqueeze(0)
    similarities = util.cos_sim(user_embedding, anime_embeddings)[0]
    average_scores_tensor = torch.tensor(anime_df["average_score"].values, dtype=torch.float32)
    top_k = torch.topk(similarities, k).indices

    recommend_animes = anime_df.iloc[top_k].copy()

    return recommend_animes

st.set_page_config(page_title="Anime Recommender for Non-Anime Fans", layout="wide")

st.title("🎌 Anime Recommendations for Non-Anime Viewers")
st.markdown("""
Welcome! This tool recommends anime shows or movies based on shows you already enjoy. 

This project explores the globalization of anime by drawing connections between anime and mainstream Western media. How do your preferences in western shows align with your taste in anime?
We use natural language processing to analyze the plot descriptions of both anime and non-anime shows, allowing us to find similarities in themes, genres, and tones.
""")

st.title('Select Shows/Movies You Enjoy')

row1, row2 = st.columns([4, 1])


with row1:
    selected_titles = st.multiselect(
        'Pick the shows or movies you enjoy:',
        options=tv_df['title'].tolist(),
        default=[]  
    )
with row2:
    num_recommendations = st.number_input(
        'Number of recommendations to generate:',
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

if not selected_titles:
    st.write("No shows/movies selected.")

if st.button("Generate Recommendations"):
    if selected_titles:
        recommended = recommend_anime(selected_titles, k=num_recommendations)
        if not recommended.empty:
            for i, row in recommended.iterrows():
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.write(f"**{row['english_title']}**")
                    st.image(row['image_url'], use_container_width=True)

                with col2:
                    innercol1, innercol2, innercol3 = st.columns([2, 3, 3])
                    with innercol1:
                        st.write("**Score:**")
                        score = row['average_score']
                        color = score_to_color(score)
                        st.markdown(f"<span style='color:{color}; font-weight:bold'>{score}</span>", unsafe_allow_html=True)
                    with innercol2:
                        st.write("**Genres:**")
                        st.write(row['genres'])
                    with innercol3:
                        st.write("**Airing Date:**")
                        st.write(f"{row['start_date']} to {row['end_date']}")

                    st.write(row['description'])

                st.markdown("---")
        else:
            st.write("Issue with generating recommendations.")
    else:
        st.write("Please select at least one title to get recommendations.")


st.markdown("---")
st.markdown("**About this project**")
st.markdown("""
This project was created as a final project for an anime course, exploring how anime's themes and genres align with global media trends. 

It uses data scraped using the AniList API and TMDB API, and is built with **Python** and **Streamlit**.
""")
