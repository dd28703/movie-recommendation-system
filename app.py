import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('movie_data.csv')
df['tags'] = df['tags'].fillna('')


tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vector = tfidf.fit_transform(df['tags']).toarray()


similarity = cosine_similarity(vector)


def recommend(movie):
    movie = movie.lower()
    if movie not in df['title'].str.lower().values:
        return ['Movie not found. Please try a different title.']
    
    index = df[df['title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]
    return [df.iloc[i[0]].title for i in movies_list]


st.set_page_config(page_title="Movie Recommendation App")
st.title("Movie Recommendation System")

user_input = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(user_input)
        for i, title in enumerate(recommendations, 1):
            st.write(f"{i}. {title}")
