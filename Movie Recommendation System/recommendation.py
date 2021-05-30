#%%
#importing the required libraries
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Reading the movies data
movies = pd.read_csv("C:/Users/KAVYA/Desktop/6th sem/miniProject/movies.csv")
# Reading the ratings data
ratings = pd.read_csv("C:/Users/KAVYA/Desktop/6th sem/miniProject/ratings.csv")
#Just taking the required columns and creating pivot matrix and filling all the nan values with 0
result = ratings.pivot(index='movieId',columns='userId',values='rating')
result.fillna(0,inplace=True)

# Counting the number of users has rated a movie
userVoted = ratings.groupby('movieId')['rating'].agg('count')
# Counting the number of movies a user has rated
moviesVoted = ratings.groupby('userId')['rating'].agg('count')

#To qualify a movie, a minimum of 10 users should have voted a movie.
#To qualify a user, a minimum of 50 movies should have voted by the user.
result = result.loc[userVoted[userVoted > 10].index,:]
result=result.loc[:,moviesVoted[moviesVoted > 50].index]

#To reduce the sparsity we use the csr_matrix function from the scipy library.
csr_data = csr_matrix(result.values)
result.reset_index(inplace=True)
#We will be using the KNN algorithm to compute similarity with cosine distance metric
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)

#train the dataset
model_knn.fit(csr_data)

#making the movie recommendation function

def movie_recommendation_function(movie_name):
    #number of movies to recommend
    recommend_movies = 10
    
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_index= movie_list.iloc[0]['movieId']
        movie_index = result[result['movieId'] == movie_index].index[0]
        distances , indices = model_knn.kneighbors(csr_data[movie_index],n_neighbors=recommend_movies+1)    
        #lambda means an inline function, as the function call implies: return the list element, using x[1] as the key.
        movie_indx_recommend = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        rec_frame = []
        for val in movie_indx_recommend:
            movie_idx = result.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            rec_frame.append({'Title':movies.iloc[idx]['title'].values[0]})
        res = pd.DataFrame(rec_frame,index=range(1,recommend_movies+1))
        movie_list=""
        for i in range(1,11):
            movie_list+=res['Title'][i]
            movie_list+=" , "
        sorted(movie_list)
        return(movie_list)
        
    else:
        return ("No movies found. Please check your input")

# res=movie_recommendation_function('Mean Girls')
# print(res)

# %%

# %%