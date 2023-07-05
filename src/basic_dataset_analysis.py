import math
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read ratings.dat into a pandas dataframe
ratings = pd.read_csv('ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')

# Print number of ratings, number of users and number of movies with
print(f'Number of ratings: {len(ratings)}')
print(f'Number of users: {len(ratings["user_id"].unique())}')
print(f'Number of movies: {len(ratings["movie_id"].unique())}')

# Plot histogram of the ratings as a whole with logarithmic y-scale
ratings['rating'].plot.hist(bins=10, title='Histogram of All Ratings', logy=True)
plt.xlabel('Rating')
plt.ylabel('Frequency of rating')
plt.savefig('hist_all_ratings.png')

# Histogram of the average rating per movie
ratings.groupby('movie_id')['rating'].mean().plot.hist(bins=50, title='Histogram of Average Movie Ratings',  logy=True)
plt.xlabel('Average Rating')
plt.ylabel('# of movies')
plt.savefig('hist_avg_movie_ratings.png')

# Histogram of the average rating per user
ratings.groupby('user_id')['rating'].mean().plot.hist(bins=50, title='Histogram of Average User Ratings',  logy=True)
plt.xlabel('Average Rating')
plt.ylabel('# of users')
plt.savefig('hist_avg_user_ratings.png')

# Heatmap of the ratings depending on the user and movie average ratings
user_avg_ratings = ratings.groupby('user_id')['rating'].mean()
movie_avg_ratings = ratings.groupby('movie_id')['rating'].mean()

ratings['user_avg_rating'] = ratings['user_id'].map(user_avg_ratings)
ratings['movie_avg_rating'] = ratings['movie_id'].map(movie_avg_ratings)
# Define the number of bins
num_bins = 25

# Create bins for 'user_avg_rating' and 'movie_avg_rating'
user_bins = pd.cut(ratings['user_avg_rating'], bins=num_bins, include_lowest=True)
movie_bins = pd.cut(ratings['movie_avg_rating'], bins=num_bins, include_lowest=True)

# Assign bins to each rating
ratings['user_bin'] = user_bins
ratings['movie_bin'] = movie_bins

# Calculate the logarithmic frequency for each combination of user_bin and movie_bin
frequency_table = ratings.groupby(['user_bin', 'movie_bin']).size().reset_index(name='frequency')
frequency_table['log_frequency'] = frequency_table['frequency'].apply(lambda x: 0 if x == 0 else round(np.log10(x)))

# Create a pivot table to reshape the data for the heatmap
pivot_table = frequency_table.pivot(index='user_bin', columns='movie_bin', values='log_frequency')

# Define the tick locations and labels for x and y axes

tick_labels = np.arange(0, 5.5, 0.5)

# Create the heatmap using seaborn
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(pivot_table.iloc[::-1], cmap='YlGnBu', annot=False, cbar_kws={'label': 'Logarithmic Frequency'}, fmt='g')
plt.xticks(tick_labels*5, tick_labels, rotation='vertical')
plt.yticks((tick_labels*5), tick_labels[::-1])
plt.xlabel('Movie\'s Average Rating')
plt.ylabel('User\'s Average Rating')
plt.title('Heatmap of frequency of ratings based on user\' and movie\'s average ratings')

# Set colorbar ticks and labels
colorbar = plt.gcf().axes[-1]
colorbar.set_yticklabels([f'$10^{int(tick)}$' for tick in colorbar.get_yticks()])

plt.savefig('heatmap_user_movie_avg_ratings.png')