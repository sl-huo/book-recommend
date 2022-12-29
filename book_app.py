
import pandas as pd
#import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

@st.cache
def load_image(onlineurl):
    image = Image.open(requests.get(onlineurl, stream=True).raw)
    return image


@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df

df_book = load_data("data/children_book_processed.csv")
df_book_item = load_data("data/children_book_collab.csv")

@st.cache
def load_matrix(path):
    matrix = sparse.load_npz(path)
    return matrix

desfull_tfidf_matrix = load_matrix('data/desfull_tfidf_matrix.npz')
book_user_matrix = load_matrix('data/book_user_matrix.npz')

### Show entered books
def show_book(title):
    title = str(title)
    if df_book['title'].str.contains(title, case=False).any():
        book_index = df_book[df_book['title'].str.contains(title, case=False)].sort_values(by='ratings_count',ascending=False).index[0]
        url = df_book.iloc[book_index]['image_url']
        st.image(url, width =180, use_column_width=False, caption=df_book.iloc[book_index]['title'])
    else:
        st.write('Sorry, book not found. Instead, below are some recommended popular books.')


### Content Based Recommendation
def content_recommender(title, vote_threshold=100):
    
    title = str(title)
    # check if title exists in dataset
    if df_book['title'].str.contains(title, case=False).any():
        # locate the book's index
        book_index = df_book[df_book['title'].str.contains(title, case=False)].sort_values(by='ratings_count',ascending=False).index[0]
        # create a book dataframe with key features
        sim_books = pd.DataFrame({'title': df_book['title'], 
                               'similarity': cosine_similarity(desfull_tfidf_matrix[book_index], desfull_tfidf_matrix).squeeze(),
                           'rating count': df_book['ratings_count'],
                           'rating': df_book['average_rating'],
                           'image_url': df_book['image_url']})
        # select top 5 books with highest similarity score
        top_books = sim_books[sim_books['rating count'] > vote_threshold].sort_values(by='similarity', ascending=False).iloc[1:6]
        return top_books
        # plot the book cover
        # cols= st.columns(5)
        # for url, title, col in zip(top_books['image_url'], top_books['title'], cols[0:]):
        #     with col:
        #         st.image(url, use_column_width=True, caption=title)   
    else:
        # if book title does not exist, recommend top rated books
        top_books = df_book[df_book['ratings_count'] > 50000].sort_values(by='average_rating', ascending=False).head(5)
        return top_books
        # plot book cover images
        # cols= st.columns(5)
        # for url, title, col in zip(top_books['image_url'], pop_df['title'], cols[0:]):
        #     with col:
        #         st.image(url, use_column_width=True, caption=title)

### User-Item Collaborative Based Recommendation
def collaborative_recommender(title, vote_threshold=100):
    
    title = str(title)
    
    if df_book_item['title'].str.contains(title, case=False).any():
        
        book_index = df_book_item[df_book_item['title'].str.contains(title, case=False)].sort_values(by='ratings_count',ascending=False).index[0]
        # calculate cosine similarity based on user-item matrix instead
        sim_item_df = pd.DataFrame({'title': df_book_item['title'], 
                           'similarity': cosine_similarity(book_user_matrix[book_index], book_user_matrix).squeeze(),
                           'rating count': df_book_item['ratings_count'],
                           'rating': df_book_item['average_rating'],
                           'image_url': df_book_item['image_url']})
        
        top_books = sim_item_df[sim_item_df['rating count'] > vote_threshold].sort_values(by='similarity', ascending=False).iloc[1:6]
        return top_books
        # cols= st.columns(5)
        # for url, title, col in zip(top_books['image_url'], top_books['title'], cols[0:]):
        #     with col:
        #         st.image(url, use_column_width=True, caption=title)
    else:
        top_books = df_book_item[df_book_item['ratings_count'] > 50000][['title', 'average_rating', 'image_url']].sort_values(by='average_rating', ascending=False).iloc[5:10]
        return top_books
        # cols= st.columns(5)
        # for url, title, col in zip(pop_df['image_url'], top_books['title'], cols[0:]):
        #     with col:
        #         st.image(url, use_column_width=True, caption=title)

### Author Based Recommendation
def author_recommender(author, vote_threshold=50):
    
    author = str(author)
    
    if df_book['author_name'].str.contains(author, case=False).any():
        df_author = df_book[df_book['author_name'].str.contains(author, case=False)]
        top_books = df_author[df_author['ratings_count'] > vote_threshold].sort_values(by='average_rating', ascending=False).head()
        cols= st.columns(5)
        for url, title, col in zip(top_books['image_url'], top_books['title'], cols[0:]):
            with col:
                st.image(url, use_column_width=True, caption=title)
    else:
        st.write('Sorry, no author found by that name. Please try again.')

### APP Presense
st.title("Children's Book Recommendation")
image = load_image('https://images.unsplash.com/photo-1472162072942-cd5147eb3902?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1769&q=80')
st.image(image, caption='Photo by Ben White on Unsplash')

def main():
    menu = ['Search by Book Title','Search by Author']
    choice = st.sidebar.selectbox("Choose recommendation methods below",menu)
    
    if choice == 'Search by Book Title':
        st.subheader('Recommend by Book Title')
        st.write('Please note that due to limited resources, this dataset does not include a comprehensive list of titles.')
        book_title = st.text_input('Enter a book you like:', 'the very hungry caterpillar')
        show_book(book_title)
        
        ### Content Based
        st.subheader('You may also like these books:\n (Recommendations based on book\'s description text)')  
        top_books_content=content_recommender(book_title)
        cols= st.columns(5)
        for url, title, col in zip(top_books_content['image_url'], top_books_content['title'], cols[0:]):
            with col:
                st.image(url, use_column_width=True, caption=title)
        ### Collaborative Based
        st.subheader('Readers also liked these books:\n (Recommendations based on reader\'s ratings)')   
        top_books_collab = collaborative_recommender(book_title)
        cols=st.columns(5)
        for url, title, col in zip(top_books_collab['image_url'], top_books_collab['title'], cols[0:]):
            with col:
                st.image(url, use_column_width=True, caption=title)
    
    else:
        ### Author Recommendation
        st.subheader('Recommend by Author')
        st.write('Please note that due to limited resources, this dataset does not include a comprehensive list of authors.')
        author_like = st.text_input('Enter an author you like:', 'Dr. Seuss')
        author_recommender(author_like)

if __name__ == '__main__':
    main()

st.write("##")
st.write('Welcome feedback and advice, please email me: shuo.2020@outlook.com')
st.write('Silei Huo | BrainStation Data Science | Vancouver')

