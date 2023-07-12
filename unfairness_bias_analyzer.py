# Path: unfairness_bias_analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

MODEL_NAME = 'GLOCAL_K'

train_samples: pd.DataFrame = pd.read_csv(f'outputs/{MODEL_NAME}/train_samples.csv')
test_samples: pd.DataFrame = pd.read_csv(f'outputs/{MODEL_NAME}/test_samples_with_preds.csv')
all_samples: pd.DataFrame = pd.concat([train_samples, test_samples])

# Calculate the average rating per user and movie
user_avg_ratings = all_samples.groupby('user_id')['rating'].mean()
movie_avg_ratings = all_samples.groupby('movie_id')['rating'].mean()

# Print number of ratings, number of users and number of movies
print(f'Number of ratings: {len(all_samples)}')
print(f'Number of users: {len(all_samples["user_id"].unique())}')
print(f'Number of movies: {len(all_samples["movie_id"].unique())}')

# How many users are in the test set but not in the training set?
print(f'Number of users in test set but not in training set: {len(test_samples[~test_samples["user_id"].isin(train_samples["user_id"])])}')

# Create directory for figures
import os
os.makedirs('figures/'+ MODEL_NAME + '/', exist_ok=True)

# Plot histogram of the ratings as a whole with logarithmic y-scale
all_samples['rating'].plot.hist(bins=10, title='Histogram of All Ratings', logy=True)
plt.xlabel('Rating')
plt.ylabel('Frequency of rating')
plt.savefig('figures/'+ MODEL_NAME + '/hist_all_ratings.png')
plt.clf()

# Histogram of the average rating per movie
all_samples.groupby('movie_id')['rating'].mean().plot.hist(bins=50, title='Histogram of Average Movie Ratings',  logy=True)
plt.xlabel('Average Rating')
plt.ylabel('# of movies')
plt.savefig('figures/'+ MODEL_NAME + '/hist_avg_movie_ratings.png')
plt.clf()

# Histogram of the average rating per user
all_samples.groupby('user_id')['rating'].mean().plot.hist(bins=50, title='Histogram of Average User Ratings',  logy=True)
plt.xlabel('Average Rating')
plt.ylabel('# of users')
plt.savefig('figures/'+ MODEL_NAME + '/hist_avg_user_ratings.png')
plt.clf()

# Heatmap of the ratings depending on the user and movie average ratings
all_samples['user_avg_rating'] = all_samples['user_id'].map(user_avg_ratings) 
all_samples['movie_avg_rating'] = all_samples['movie_id'].map(movie_avg_ratings)

# Define the number of bins
num_bins = 10

# Create bins for 'user_avg_rating' and 'movie_avg_rating'
user_bins = pd.cut(all_samples['user_avg_rating'], bins=num_bins, include_lowest=True)
movie_bins = pd.cut(all_samples['movie_avg_rating'], bins=num_bins, include_lowest=True)

# Assign bins to each rating
all_samples['user_bin'] = user_bins
all_samples['movie_bin'] = movie_bins

# Calculate the logarithmic frequency for each combination of user_bin and movie_bin
frequency_table = all_samples.groupby(['user_bin', 'movie_bin']).size().reset_index(name='frequency')
frequency_table['log_frequency'] = frequency_table['frequency'].apply(lambda x: 0 if x == 0 else round(np.log10(x)))

# Create a pivot table to reshape the data for the heatmap
pivot_table = frequency_table.pivot(index='user_bin', columns='movie_bin', values='log_frequency')

# Define the tick locations and labels for x and y axes

tick_labels = np.arange(0, 5.5, 0.5)

# Create the heatmap using seaborn
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(pivot_table.iloc[::-1], cmap='YlGnBu', annot=False, cbar_kws={'label': 'Logarithmic Frequency'}, fmt='g')
plt.xticks(tick_labels*2, tick_labels, rotation='vertical')
plt.yticks((tick_labels*2), tick_labels[::-1])
plt.xlabel('Movie\'s Average Rating')
plt.ylabel('User\'s Average Rating')
plt.title('Heatmap of frequency of ratings based on user\' and movie\'s average ratings')

# Set colorbar ticks and labels
colorbar = plt.gcf().axes[-1]
colorbar.set_yticklabels([f'$10^{int(tick)}$' for tick in colorbar.get_yticks()])

plt.savefig('figures/'+ MODEL_NAME + '/ratings_distribution_heatmap.png')
plt.clf()

### UNFAIRNESS BIAS IN TEST SET ###

# Add the user and movie average ratings to the test set
test_samples['user_avg_rating'] = test_samples['user_id'].map(user_avg_ratings)
test_samples['movie_avg_rating'] = test_samples['movie_id'].map(movie_avg_ratings)

# Create bins for 'user_avg_rating' and 'movie_avg_rating'
user_bins = pd.cut(test_samples['user_avg_rating'], bins=num_bins, include_lowest=True)
movie_bins = pd.cut(test_samples['movie_avg_rating'], bins=num_bins, include_lowest=True)

# Assign bins to each rating
test_samples['user_bin'] = user_bins
test_samples['movie_bin'] = movie_bins

test_samples['error'] = test_samples['rating'] - test_samples['prediction']

# Calculate the RMSE for each combination of user_bin and movie_bin
rmse_table = test_samples.groupby(['user_bin', 'movie_bin'])['error'].apply(lambda x: np.sqrt(np.mean(x**2))).reset_index(name='rmse')

# Create a pivot table to reshape the data for the heatmap
pivot_table = rmse_table.pivot(index='user_bin', columns='movie_bin', values='rmse')

# Create the heatmap using seaborn
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(pivot_table.iloc[::-1], cmap='YlGnBu', annot=False, cbar_kws={'label': 'RMSE'}, fmt='g')
plt.xticks(tick_labels*2, tick_labels, rotation='vertical')
plt.yticks((tick_labels*2), tick_labels[::-1])
plt.xlabel('Movie\'s Average Rating')
plt.ylabel('User\'s Average Rating')
plt.title('Heatmap of RMSE based on user\' and movie\'s average ratings')

plt.savefig('figures/'+ MODEL_NAME + '/rmse_heatmap.png')
plt.clf()

### 2d heatmap of rating vs prediction ###
# Create bins for ¡rating' and 'prediction'	

num_bins = 10

rating_bins = pd.cut(test_samples['rating'], bins=num_bins, include_lowest=True)
prediction_bins = pd.cut(test_samples['prediction'], bins=num_bins, include_lowest=True)

# Assign bins to each rating
test_samples['rating_bin'] = rating_bins
test_samples['prediction_bin'] = prediction_bins

# Calculate the logarithmic frequency for each combination of user_bin and movie_bin
frequency_table = test_samples.groupby(['rating_bin', 'prediction_bin']).size().reset_index(name='frequency')
frequency_table['log_frequency'] = frequency_table['frequency'].apply(lambda x: 0 if x == 0 else round(np.log10(x)))

# Create a pivot table to reshape the data for the heatmap
pivot_table = frequency_table.pivot(index='rating_bin', columns='prediction_bin', values='log_frequency')

# Define the tick locations and labels for x and y axes

tick_labels = np.arange(0, 5.5, 0.5)

# Create the heatmap using seaborn
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(pivot_table.iloc[::-1], cmap='YlGnBu', annot=False, cbar_kws={'label': 'Logarithmic Frequency'}, fmt='g')
plt.xticks(tick_labels*5, tick_labels, rotation='vertical')
plt.yticks((tick_labels*5), tick_labels[::-1])
plt.xlabel('Prediction')
plt.ylabel('Rating')
plt.title('Heatmap of frequency of ratings based on rating and prediction')

# Set colorbar ticks and labels
colorbar = plt.gcf().axes[-1]
colorbar.set_yticklabels([f'$10^{int(tick)}$' for tick in colorbar.get_yticks()])
plt.savefig('figures/'+ MODEL_NAME + '/ratings_vs_predictions_heatmap.png')
plt.clf()
