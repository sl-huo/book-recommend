
import pandas as pd
import requests
from PIL import Image
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

@st.cache
def load_data_index(path):
    df = pd.read_csv(path, index_col=0)
    return df

@st.cache
def load_matrix(path):
    matrix = sparse.load_npz(path)
    return matrix

######################################################################
### Load dataset and matrix
df_book = load_data("data/children_book_processed.csv")
df_book_item = load_data("data/children_book_collab.csv")
df_svd_recommend_book = load_data_index('data/children_user_recommendation.csv')
df_svd_toprated = load_data('data/children_user_toprated.csv')
desfull_tfidf_matrix = load_matrix('data/desfull_tfidf_matrix.npz')
book_user_matrix = load_matrix('data/book_user_matrix.npz')

######################################################################
### Show entered books
def show_book(title):
    title = str(title)
    if df_book['title'].str.contains(title, case=False).any():
        book_index = df_book[df_book['title'].str.contains(title, case=False)].sort_values(by='ratings_count',ascending=False).index[0]
        url = df_book.iloc[book_index]['image_url']
        st.image(url, width =150, use_column_width=False, caption=df_book.iloc[book_index]['title'])
    else:
        st.write('Sorry, book not found. Instead, below are some recommended popular books.')

######################################################################
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
        
    else:
        # if book title does not exist, recommend top rated books
        top_books = df_book[df_book['ratings_count'] > 50000].sort_values(by='average_rating', ascending=False).head(5)
        return top_books
        
######################################################################
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
       
    else:
        top_books = df_book_item[df_book_item['ratings_count'] > 50000][['title', 'average_rating', 'image_url']].sort_values(by='average_rating', ascending=False).iloc[5:10]
        return top_books
        

######################################################################
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

######################################################################
### User Based Recommendation (FunkSVD)
def user_recommender(user):
    user=int(user)
    if user in df_svd_recommend_book.index:
        book_list = list(df_svd_recommend_book.loc[user])
        recommend_df = df_book.loc[df_book['book_id'].isin(book_list)]
        recommend_books = recommend_df[['title', 'image_url']].sample(n=5)
        cols= st.columns(5)
        for url, title, col in zip(recommend_books['image_url'], recommend_books['title'], cols[0:]):
            with col:
                st.image(url, use_column_width=True, caption=title)
    else:
        st.write('Sorry, no user found by that id. Please try again.')

def user_profile(user):
    if user in df_svd_toprated['user'].unique():
        book_list = list(df_svd_toprated[df_svd_toprated['user']==user]['book_id'])
        toprated_books = df_book.loc[df_book['book_id'].isin(book_list)]
        cols=st.columns(10)
        for url, title, col in zip(toprated_books['image_url'], toprated_books['title'], cols[0:]):
            with col:
                st.image(url, use_column_width=True)
    else:
        st.write('Sorry, no user found by that id. Please try again.')

######################################################################
### APP Presense
st.title("Children's Book Recommendation\n 英语儿童书籍推荐")
image = load_image('https://images.unsplash.com/photo-1472162072942-cd5147eb3902?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1769&q=80')
st.image(image, caption='Photo by Ben White on Unsplash')

tab1, tab2, tab3 = st.tabs(["Search by Book Title\n 按书名搜索", "Search by Author\n 按作者搜索", "Recommend to Existing Users\n 读者推荐"])

with tab1:
    st.header('Recommend by Book Title')
    st.caption('Please note that due to limited resources, this dataset does not include a comprehensive list of titles.')
    st.subheader('Enter a book you like:\n请输入你一本你喜欢的书名:')
    book_title = st.text_input('Enter a book you like:', 'the very hungry caterpillar', label_visibility="collapsed")
    show_book(book_title)
        
    ### Content Based
    st.subheader('You may also like these books:\n你可能会喜欢以下类似书籍:')  
    top_books_content=content_recommender(book_title)
    cols= st.columns(5)
    for url, title, col in zip(top_books_content['image_url'], top_books_content['title'], cols[0:]):
        with col:
            st.image(url, use_column_width=True, caption=title)
    st.markdown(':bulb: Above recommendations based on book\'s description text - Content Based Model')
    ### Collaborative Based
    st.subheader('Readers also liked these books:\n喜欢这本的读者还喜欢以下书籍:')
    top_books_collab = collaborative_recommender(book_title)
    cols=st.columns(5)
    for url, title, col in zip(top_books_collab['image_url'], top_books_collab['title'], cols[0:]):
        with col:
            st.image(url, use_column_width=True, caption=title)
    st.markdown(':bulb: Above recommendations based on reader\'s ratings - Collaborative Filtering Model')

with tab2:
    ### Author Recommendation
    st.header('Recommend by Author')
    st.caption('Please note that due to limited resources, this dataset does not include a comprehensive list of authors.')
    st.subheader('Enter an author you like:\n请输入一位你喜欢的作者:')
    author_like = st.text_input('Enter an author you like:', 'Eric Carle', label_visibility="collapsed")
    st.markdown(f'You may like these books from {author_like}, 你可能会喜欢{author_like}的这些书籍:')
    author_recommender(author_like)

with tab3:
    ### User Recommendation
    st.header('Recommend to Existing Users')
    st.markdown('Recommendations based on user\'s current rating profile by using FunkSVD')
    st.subheader('Please enter your user id:\n请输入你的用户号:')
    st.caption('Please note that due to limited resources, here only shows a subsample. User ID is adjusted to numbers from 1 to 3300.')
    user_id = st.number_input('', min_value=1, max_value=3342, label_visibility="collapsed")
    st.subheader('You may also like these books:\n你可能会喜欢这些书籍:')
    user_recommender(str(user_id))
    st.subheader('Your rated books:\n你已阅读并喜欢的书籍:')
    user_profile(user_id)

st.write('''<style>

[data-testid="column"] {
    width: calc(20% - 1rem) !important;
    flex: 1 1 calc(20% - 1rem) !important;
    min-width: calc(20% - 1rem) !important;
}
</style>''', unsafe_allow_html=True)



st.markdown("""---""")
st.caption('Feedback and advice is welcome, please email me: shuo.2020@outlook.com')
st.caption('Silei Huo | BrainStation Data Science | Vancouver | 2022')

