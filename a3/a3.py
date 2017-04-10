'''
 Created on Oct 29, 2016

@author: PRATIK SHAH
'''

# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    genres=movies.genres
    store = []
    for value in genres:
        store.append(tokenize_string(value))
    movies["tokens"]=store
    return movies
    ###TODO
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    df=dict()
    df_cal=dict()
    tfi=dict()
    vocab=dict()
    index=0
    df_unique_movie_id=set()
    df_unique_movie=[]
    dict_list=dict()
    dict_tokens_index=dict()
    unique_genres=set()
    flag=1
    totalCount=0.0
    index_key=0
    value_old=""
    for genres in movies["tokens"].tolist():
        if(movies.movieId[index] not in df_unique_movie_id ):
            movie_id=movies.movieId[index]
            df_unique_movie_id.add(movies.movieId[index])
            totalCount=totalCount+1.0
            settings={}
            flag=0
            settings["max_count"]=1
            unique_genres.clear()
            genres=sorted(genres)
            for value in genres:
                if value not in settings:
                    settings[value]=1
                if(flag==1): 
                    if(value_old==value):
                        settings["max_count"]= settings["max_count"]+1.0
                        settings[value]=settings[value]+1
                if(value not in df and value not in unique_genres):
                    df[value]=1
                    flag=1
                    df_unique_movie.append(value)
                    unique_genres.add(value)
                else:
                    if(value not in unique_genres):
                        flag=1
                        df[value]=df[value]+1
                        unique_genres.add(value)
                value_old=value    
            dict_list[index_key]=settings
            dict_tokens_index[index_key]=movies.tokens[index_key]
            index_key=index_key+1
        index=index+1
       
    ###TODO
    i=0
    df_unique_movie_genre=sorted(df_unique_movie)
    for values in df_unique_movie_genre:
        vocab[values]=i
        i=i+1
        df_cal[values]=math.log10(totalCount/df[values])
    N=index
    i=0
    data = []
    row = []
    column=[]
    maxcount=1
    store = []
    rowcounter=0
    key=0
    df_cal=sorted(df_cal.items())
    while(True):
        data = []
        row = []
        column=[]
    #for movieid in movies.movieId.tolist():
        if(key>=index_key):
            break
        termsvalue=dict_list[key]
        col=0
        key=key+1
        for genrename,val in df_cal:
            flag_add=0  
        #    for keyval,term in value.items():
            if(genrename  in dict_tokens_index[key-1]):
                
                terms=termsvalue[genrename]      
                tfidf=(terms/termsvalue["max_count"])*val
                column.append(vocab[genrename])
                data.append(tfidf)
                flag_add=1
                row.append(0)
            if(flag_add==0):
                row.append(0)      
                column.append(vocab[genrename])
                data.append(0)
            #col=col+1
        #csr_matrix((data, (row, column)))
        #movies.set_value(key-1,"features",csr_matrix((data, (row, column))))

        store.append(csr_matrix((data, (row, column))))           
        #movies["features"]=csr_matrix((data, (row, column)))
        rowcounter=rowcounter+1
    movies["features"]=store
    return movies,vocab
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    norm_multiplication = 0.0
    cosine = 0.0
    result = 0.0
    j=0
    size=a._shape[0]
    while j<=a._shape[1]-1:
                if(a[0,j]==0 or b[0,j]==0):
                    result=result+0
                else:
                    result = result + a[0,j]* b[0,j]
                j=j+1

    norma=norm(a)
    normb=norm(b)
    if(norma*normb!=0):
        cosine=result/(norma*normb)
    else:
        cosine=0.0;
    return cosine
    pass


def norm(vector):
    sumval = 0.0
    j=0 
    while j<vector._shape[1]:
        if(vector[0,j]==0):
            sumval=sumval+0
        else:
            sumval = sumval + (vector[0,j]**2)
        j=j+1
    euclidean_norm = math.sqrt(sumval) 
    return euclidean_norm
    

def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    dict_list=dict()
    dict_computed_cosine_storage=dict()
    indexuid=-1
    movieId_test=ratings_test.movieId.tolist()
    movieId_train=ratings_train.movieId.tolist()
    result=0.0

    for uid in ratings_test.userId:

        indexuid=indexuid+1
        #store=[]
        ratingmovieid=movieId_test[indexuid]
        a=movies.loc[movies.movieId==ratingmovieid].features.values[0]
        list_val=[]
        list_val=ratings_train.loc[ratings_train.userId==uid]
        index_value=-1
        Total=0.0
        cosineadd=0.0
        avg=0.0
        dict_computed_cosine_storage=dict()
        for movieId in list_val.movieId.values :
            index_value=index_value+1
            b=movies.loc[movies.movieId==movieId]
            valuegenres=b.genres.values[0]
            if valuegenres not in dict_computed_cosine_storage:
                result=cosine_sim(a,b.features.values[0])
                dict_computed_cosine_storage[valuegenres]=result
            else:
                result=dict_computed_cosine_storage[valuegenres]
            if(result>=0):
               
                cosineadd=cosineadd+result
                
                rate=list_val.rating.values[index_value]
                Total=(result*rate)+Total
                #index_value=index_value+1
                avg=avg+rate
            
            #break;
            #    dict_list[movieId_train[indexuserids]]=cosine_sim(a.features.values[0],b.features.values[0])
        #print (cosineadd)
        if(cosineadd==0.0):
           dict_list[indexuid]=avg/(len(list_val.movieId.values ))
        if(cosineadd!=0):
            dict_list[indexuid] =(Total)/cosineadd  

    tokens=[]    
    for key,value in    dict_list.items():
        tokens.append(value) 
        #print(value)  
    return np.array(tokens)
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
