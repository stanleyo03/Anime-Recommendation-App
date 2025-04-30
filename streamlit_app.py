import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch


# embedding1 = model.encode("Plot of non-anime show")
# embedding2 = model.encode("Plot of anime show")

# similarity = util.cos_sim(embedding1, embedding2)

model = SentenceTransformer('all-MiniLM-L6-v2')

tv_df = pd.read_csv("data/non_anime.csv")
anime_df = pd.read_csv("data/anime_metadata.csv")

def recommend_anime(user_inputs):
    user_show_descriptions = tv_df[tv_df['title'].isin(user_inputs)]['overview'].tolist()

    anime_embeddings = model.encode(anime_df['description'].tolist(), convert_to_tensor=True)

    user_embeddings = model.encode(user_show_descriptions, convert_to_tensor=True)

    user_embedding = torch.mean(user_embeddings, dim=0).unsqueeze(0)
    similarities = util.cos_sim(user_embedding, anime_embeddings)[0]
    average_scores_tensor = torch.tensor(anime_df["average_score"].values, dtype=torch.float32)
    top_k = torch.topk(similarities * average_scores_tensor, k=5).indices

    recommend_animes = anime_df.iloc[top_k].copy()

    return recommend_animes

# App layout
st.set_page_config(page_title="Anime Recommender for Non-Anime Fans", layout="wide")

st.title("🎌 Anime Recommendations for Non-Anime Viewers")
st.markdown("""
Welcome! This tool recommends **anime shows or movies** based on the **non-anime content you already enjoy**.

This project explores the globalization of anime by drawing connections between anime and mainstream Western media.
""")

# Create a Streamlit selection component for users to choose multiple titles
st.title('Select Shows/Movies You Enjoy')

# Use the multiselect widget to allow multiple selections
selected_titles = st.multiselect(
    'Pick the shows or movies you enjoy:',
    options=tv_df['title'].tolist(),
    default=[]  # Default selection, can be customized
)

# Show selected titles
if selected_titles:
    st.write(f"You selected {len(selected_titles)} titles:")
    for title in selected_titles:
        col1, col2 = st.columns([2, 3])  # Adjust column widths as needed

        with col1:
            st.write(f"**{title}**")
        with col2: 
            st.write(tv_df[tv_df['title'] == title]['overview'].values[0])
else:
    st.write("No shows/movies selected.")

if st.button("Generate Recommendations"):
    if selected_titles:
        recommended = recommend_anime(selected_titles)
        if not recommended.empty:
            for i, row in recommended.iterrows():
                # Create a row with 3 columns
                col1, col2 = st.columns([2, 3])  # Adjust column widths as needed

                # Column 1: Image
                with col1:
                    st.write(f"**{row['english_title']}**")
                    st.image(row['image_url'], use_container_width=True)

                # Column 2: Title and Description
                with col2:
                    st.write(row['description'])

                st.markdown("---")  # Add a separator between rows
        else:
            st.write("Issue with generating recommendations.")
    else:
        st.write("Please select at least one title to get recommendations.")



# # Input form
# with st.form("recommendation_form"):
#     st.subheader("Tell us what you watch 🎥📺")
#     user_shows = st.text_area("Enter a list of non-anime films or TV shows you enjoy (separated by commas)", 
#                               placeholder="e.g. Breaking Bad, The Witcher, Black Mirror")

#     submit = st.form_submit_button("Recommend me anime!")

# # When the form is submitted
# if submit and user_shows:
#     st.subheader("🔍 Recommendations Based on Your Input")
    
#     # Convert user input into list
#     user_inputs = [s.strip() for s in user_shows.split(",") if s.strip()]
    
#     # Generate recommendations (replace with your own function)
#     recommendations = recommend_anime(user_inputs)
    
#     # Display results
#     st.dataframe(recommendations)
#     st.markdown("✍️ *These recommendations are based on shared themes, tone, or genre similarities.*")

st.markdown("---")
st.markdown("👩‍💻 **About this project**")
st.markdown("""
This project was created as a final project for an anime course, exploring how anime's themes and genres align with global media trends. 

It uses data from **IMDb** and **MyAnimeList**, and is built with **Python** and **Streamlit**.
""")
