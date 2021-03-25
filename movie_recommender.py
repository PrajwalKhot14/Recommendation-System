import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

def combined_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director'] 

df = pd.read_csv(r'movie_dataset.csv')
features = ['keywords', 'cast', 'genres', 'director']
for feature in features:
    df[feature] = df[feature].fillna(' ')

df['combined_features'] = df.apply(combined_features, axis = 1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(count_matrix)
#print(similarity_score)

movie = input("Movie Name: ")
movie_index = get_index_from_title(movie)
similar_movies = list(enumerate(cosine_sim[movie_index]))
sorted_movies = sorted(similar_movies, key= lambda x:x[1], reverse=True)

print("Recommended Movies:")
i = 0
for index in sorted_movies:
    print(get_title_from_index(index[0]), index[1])
    i = i + 1
    if i>50:
        break
    
