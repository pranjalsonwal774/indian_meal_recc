import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
import os

# ------------------------ Set Page Configuration ------------------------
st.set_page_config(page_title="Meal Recommender", layout="wide")

# ------------------------ Background Image Function ------------------------
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

# Set background image (make sure meal.jpg is in the same directory)
set_background("image.png")

# ------------------------ Load and Process Dataset ------------------------
data = pd.read_csv("recipeex001.csv")

vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['Ingredients'])

scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['TotalTimeInMins']])

X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)

# ------------------------ Recommendation Function ------------------------
def recommend_recipes(TotalTimeInMins, Ingredients):
    input_scaled = scaler.transform(np.array([[TotalTimeInMins]]))
    input_ingredients = vectorizer.transform([Ingredients])
    input_combined = np.hstack([input_scaled, input_ingredients.toarray()])
    distances, indices = knn.kneighbors(input_combined)
    return data.iloc[indices[0]][['Name', 'Ingredients', 'img_url']]

# ------------------------ User Interface ------------------------
st.markdown("<h1 style='text-align: center;'>🍽️ Meal Recommendation System</h1>", unsafe_allow_html=True)

# Inject CSS for blurred container
with st.container():
    st.markdown("""
    <style>
    .blur-container {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.form("recommendation_form"):
        col1, col2 = st.columns(2)
        with col1:
            TotalTimeInMins = st.number_input("Preparation Time (in minutes)", min_value=1, step=1)
        with col2:
            Ingredients = st.text_input("Ingredients You Have")

        submit = st.form_submit_button("Get Recommendations")

# ------------------------ Display Recommendations ------------------------
if submit and Ingredients.strip():
    recs = recommend_recipes(TotalTimeInMins, Ingredients)

    st.subheader("Recommended Meals 🍛")
    cols = st.columns(3)
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 3]:
            img_path = row['img_url']
            try:
                if os.path.isfile(img_path):  # Local image
                    st.image(img_path, use_container_width=True)
                else:  # URL image
                    st.image(img_path, use_container_width=True)
            except:
                st.warning("⚠️ Image not available.")
            st.markdown(f"**{row['Name']}**")
            st.caption(row['Ingredients'][:70] + '...' if len(row['Ingredients']) > 70 else row['Ingredients'])
