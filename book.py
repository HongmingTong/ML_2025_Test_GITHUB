import pandas as pd
df = pd.read_csv('books2.csv')
min_vote=275000
mean_vote=df['ratings_count'].mean()
df['weighted_rank']=(df['ratings_count']/(df['ratings_count']+min_vote))*df['average_rating']+(min_vote/(df['ratings_count']+min_vote))*mean_vote
recommendations = df.sort_values(by='weighted_rank', ascending=False)
recommendations = recommendations[['title', 'ratings_count', 'average_rating', 'weighted_rank']].head(5)
recommendations = pd.DataFrame(recommendations.values, columns=recommendations.columns)
print('Popularity-based Recommender')
print('The top 5 books recommended using weighted ranking similar to IMDB rating is:')
print(recommendations)
print('----------------------------------------')
print('')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
df['authors'] = df['authors'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['authors'])

from sklearn.metrics.pairwise import cosine_distances
distance_matrix = cosine_distances(tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


def ContentBasedRecommender(title, indices, distance_matrix):
    id_ = indices[title]
    distances = list(enumerate(distance_matrix[id_]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    distances = distances[1:6]
    rec = [distance[0] for distance in distances]
    print(df['title'].iloc[rec])
    return df['title'].iloc[rec]
print('Content-based Recommender')
user_entry=input('List your favorite book from the dataset and see what other books will be recommended ')
print('The top 5 recommended book for your selection is:')
ContentBasedRecommender(user_entry, indices, distance_matrix)