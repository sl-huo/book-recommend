
import pandas as pd
#import os
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

st.title("Children's Book Recommendation")

@st.cache
def load_image(onlineurl):
    image = Image.open(requests.get(onlineurl, stream=True).raw)
    return image

image = load_image('https://images.unsplash.com/photo-1472162072942-cd5147eb3902?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1769&q=80')
st.image(image, caption='Photo by Ben White on Unsplash')

@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df

df_book = load_data("data/children_book_processed.csv")
df_book_item = load_data("data/children_book_collab.csv")

@st.cache
def content_similarity():
    tfidf_matrix = sparse.load_npz('data/des_tfidf_matrix.npz')
    similarities_tfidf = cosine_similarity(tfidf_matrix, dense_output=False)
    return similarities_tfidf
# Content similarity
similarities_tfidf = content_similarity()

@st.cache
def collab_similarity():
    book_user_matrix = sparse.load_npz('data/book_user_matrix.npz')
    similarities_item = cosine_similarity(book_user_matrix, dense_output=False)
    return similarities_item
# Collaborative similarity
similarities_item = collab_similarity()

# Content Based Function
def content_recommender(title, vote_threshold=100):
    
    title = str(title)
    
    if df_book['title'].str.contains(title, case=False).any():
        
        book_index = df_book[df_book['title'].str.contains(title, case=False)].sort_values(by='ratings_count',ascending=False).index[0]
        
        sim_books = pd.DataFrame({'book': df_book['title'], 
                               'similarity': np.array(similarities_tfidf[book_index, :].todense()).squeeze(),
                           'rating count': df_book['ratings_count'],
                           'rating': df_book['average_rating'],
                           'url': df_book['image_url']})
        
        top_books = sim_books[sim_books['rating count'] > vote_threshold].sort_values(by='similarity', ascending=False).head(5)
        
        fig, axs = plt.subplots(1, 5,figsize=(12,3))
        # fig.suptitle('You may also like these books', size = 12, color='indigo')
        
        for i in range(len(top_books['book'].tolist())):
            url = top_books.iloc[i]['url']
            im = Image.open(requests.get(url, stream=True).raw)
            axs[i].imshow(im)
            axs[i].axis("off")
            axs[i].set_title('Rating: {}'.format(top_books.iloc[i]['rating']),y=-0.18,color="indigo",fontsize=10)
    else:
        pop_df = df_book[df_book['ratings_count'] > 50000][['title', 'average_rating', 'image_url']].sort_values(by='average_rating', ascending=False).head(5)
        
        fig, axs = plt.subplots(1, 5,figsize=(12,3))
        fig.suptitle('Can not find the book, please check spelling. \nRecommend below popular books for you:', size = 12, color='indigo')
        
        for i in range(len(pop_df['title'].tolist())):
            url = pop_df.iloc[i]['image_url']
            im = Image.open(requests.get(url, stream=True).raw)
            axs[i].imshow(im)
            axs[i].axis("off")
            axs[i].set_title('Rating: {}'.format(pop_df.iloc[i]['average_rating']),y=-0.18,color="indigo",fontsize=10)
            

# User-Item Based Function
def collaborative_recommender(title, vote_threshold=100):
    
    title = str(title)
    
    if df_book_item['title'].str.contains(title, case=False).any():
        
        book_index = df_book_item[df_book_item['title'].str.contains(title, case=False)].sort_values(by='ratings_count',ascending=False).index[0]
        
        sim_item_df = pd.DataFrame({'book': df_book_item['title'], 
                           'similarity': np.array(similarities_item[book_index, :].todense()).squeeze(),
                           'rating count': df_book_item['ratings_count'],
                           'rating': df_book_item['average_rating'],
                           'url': df_book_item['image_url']})
        # Get the top 5 books with > 100 votes
        top_books = sim_item_df[sim_item_df['rating count'] > vote_threshold].sort_values(by='similarity', ascending=False).head(5)

        fig, axs = plt.subplots(1, 5,figsize=(12,3))
        # fig.suptitle('Readers also liked', size = 12, color='indigo')
        
        for i in range(len(top_books['book'].tolist())):
            url = top_books.iloc[i]['url']
            im = Image.open(requests.get(url, stream=True).raw)
            axs[i].imshow(im)
            axs[i].axis("off")
            axs[i].set_title('Rating: {}'.format(top_books.iloc[i]['rating']),y=-0.18,color="indigo",fontsize=10)
    else:
        pop_df = df_book_item[df_book_item['ratings_count'] > 50000][['title', 'average_rating', 'image_url']].sort_values(by='average_rating', ascending=False).head(5)
        
        fig, axs = plt.subplots(1, 5,figsize=(12,3))
        fig.suptitle('Can not find the book, please check spelling. \nRecommend below popular books for you:', size = 12, color='indigo')
        
        for i in range(len(pop_df['title'].tolist())):
            url = pop_df.iloc[i]['image_url']
            im = Image.open(requests.get(url, stream=True).raw)
            axs[i].imshow(im)
            axs[i].axis("off")
            axs[i].set_title('Rating: {}'.format(pop_df.iloc[i]['average_rating']),y=-0.18,color="indigo",fontsize=10)
            


title = st.text_input('Enter the book you like', 'the very hungry caterpillar')

st.set_option('deprecation.showPyplotGlobalUse', False)
## Content Based
st.subheader('You may also like these books:')  
fig_content = content_recommender(title)
st.pyplot(fig_content)

## User Based
st.subheader('Readers also liked these books:')   
fig_collab = collaborative_recommender(title)
st.pyplot(fig_collab)


st.write('Welcome feedback and advice, please email me: shuo.2020@outlook.com')

st.write('Silei Huo | BrainStation Data Science | Vancouver')
  
