#import libraries
import numpy as np
import pandas as pd
#visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('data\movierecommend.txt', sep='\t', names=column_names)
df.head()
movie_titles = pd.read_csv('data\Movie_Id_Titles.txt')
movie_titles.head()
df = pd.merge(df,movie_titles,on='item_id')
df.head()
sns.set_style('white')
matplotlib inline
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
df.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5);
#Recommending similar movies

# Pivot the data to create a user-item matrix
ratings_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Normalize the ratings matrix
scaler = MinMaxScaler()
ratings_matrix_scaled = scaler.fit_transform(ratings_matrix)

# Create a dictionary mapping item IDs to their titles
item_titles = dict(zip(df['item_id'], df['title']))

# Factorize the ratings matrix using truncated SVD
svd = TruncatedSVD(n_components=50, random_state=42)
ratings_matrix_svd = svd.fit_transform(ratings_matrix_scaled)

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(ratings_matrix_svd)
# Function to recommend movies based on user input
def recommend_movies(user_id, num_recommendations):
    # Get the cosine similarity scores for the user
    user_cosine_sim = cosine_sim[user_id-1]

    # Get the indices of the top recommended movies
    top_movies_idx = user_cosine_sim.argsort()[::-1][:num_recommendations]

    # Get the movie IDs of the top recommended movies
    recommendations = list(ratings_matrix.columns[top_movies_idx])

    return recommendations

recommendations = recommend_movies(1, 10)
recommended_movie = []
for i in recommendations:
    recommended_movie.append(item_titles[i])
    recommended_movie = pd.DataFrame(recommended_movie, columns=['Recommended Movies'])
    recommended_movie