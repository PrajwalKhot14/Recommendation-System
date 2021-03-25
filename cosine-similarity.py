from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

text = ["London Paris London", "Paris Paris London"]
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)

similarity_score = cosine_similarity(count_matrix)
print(similarity_score)