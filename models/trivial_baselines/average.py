import os
import pandas as pd
import numpy as np

# Read MovieLens data
data = pd.read_csv('data/ml-1m/ratings.dat', sep='::', engine='python', header=None)

# Rename columns
data.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

# Split 90-10 train-test
msk = np.random.rand(len(data)) < 0.9
train_df = data[msk]
test_df = data[~msk]

# Compute average rating per user and per movie
user_avg_ratings = train_df.groupby('user_id')['rating'].mean()
movie_avg_ratings = train_df.groupby('movie_id')['rating'].mean()

# Compute the average rating in the training set

train_avg_rating = train_df['rating'].mean()

# BASELINE 1: Average global rating

# Compute the RMSE on the test set
rmse = ((test_df['rating'] - train_avg_rating) ** 2).mean() ** 0.5
print(f'RMSE with global average: {rmse:.3f}')

# BASELINE 2: Average user rating

# Compute the RMSE on the test set
rmse = ((test_df['rating'] - test_df['user_id'].map(user_avg_ratings)) ** 2).mean() ** 0.5
print(f'RMSE with user average: {rmse:.3f}')

# BASELINE 3: Average movie rating

# Compute the RMSE on the test set
rmse = ((test_df['rating'] - test_df['movie_id'].map(movie_avg_ratings)) ** 2).mean() ** 0.5
print(f'RMSE with movie average: {rmse:.3f}')

# BASELINE 4: Average of average user and average movie ratings

# Compute the RMSE on the test set
rmse = ((test_df['rating'] - (test_df['user_id'].map(user_avg_ratings) + test_df['movie_id'].map(movie_avg_ratings)) / 2) ** 2).mean() ** 0.5
print(f'RMSE with user-movie average: {rmse:.3f}')

# Save the train and test sets
os.makedirs('outputs/AVG/', exist_ok=True)

train_df.to_csv('outputs/AVG/train_samples.csv', index=False)
test_df['pred'] = (test_df['user_id'].map(user_avg_ratings) + test_df['movie_id'].map(movie_avg_ratings)) / 2
test_df.to_csv('outputs/AVG/test_samples_with_predictions.csv', index=False)
