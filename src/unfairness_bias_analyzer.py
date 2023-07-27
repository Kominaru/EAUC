# Path: unfairness_bias_analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

MODEL_NAME = 'MF'

train_samples: pd.DataFrame = pd.read_csv(f'outputs/{MODEL_NAME}/train_samples.csv')
test_samples: pd.DataFrame = pd.read_csv(f'outputs/{MODEL_NAME}/test_samples_with_predictions.csv')
all_samples: pd.DataFrame = pd.concat([train_samples, test_samples])

# Print number of ratings, number of users and number of movies
print(f'Number of ratings: {len(all_samples)}')
print(f'Number of users: {len(all_samples["user_id"].unique())}')
print(f'Number of movies: {len(all_samples["movie_id"].unique())}')

# Maximum and mininum ratings
print(f'Maximum rating: {all_samples["rating"].max()}')
print(f'Minimum rating: {all_samples["rating"].min()}')

# Maximum and mininum predictions
print(f'Maximum prediction: {all_samples["pred"].max()}')
print(f'Minimum prediction: {all_samples["pred"].min()}')

# How many users are in the test set but not in the training set?
print(f'Number of users in test set but not in training set: {len(test_samples[~test_samples["user_id"].isin(train_samples["user_id"])])}')

train_tuples = train_samples[['user_id', 'movie_id']].apply(tuple, axis=1)
test_tuples = test_samples[['user_id', 'movie_id']].apply(tuple, axis=1)

print('Repeated samples: ', test_tuples.isin(train_tuples).sum())

# Clamp predictions to the range [1, 5]
test_samples['pred'] = test_samples['pred'].clip(1, 5)


# Brief check of the RMSE to ensure we're reproducing the results from the paper
print("Test RMSE: ", np.sqrt(np.mean((test_samples['rating'] - test_samples['pred'])**2)))

# Create directory for figures
import os
os.makedirs('figures/'+ MODEL_NAME + '/', exist_ok=True)

from figure_scripts.basic_dataset_statistics import plot_basic_dataset_statistics_figures
plot_basic_dataset_statistics_figures(all_samples.copy(), train_samples.copy(), MODEL_NAME)

from figure_scripts.all_ratings_2dheatmaps import all_ratings_2dheatmap
all_ratings_2dheatmap(train_samples.copy(), test_samples.copy(), MODEL_NAME, bin_interval=0.5)

# For each rating in the test set, plot a 2d plot of the rmse depending on the user and movie average ratings, and a 
# 2d plot of the frequency of ratings depending on the user and movie average ratings
# and organize the plots in a grid

from figure_scripts.rating_preds_conf_matrix import plot_2dheatmap_ratings_vs_preds
plot_2dheatmap_ratings_vs_preds(test_samples.copy(), MODEL_NAME, preds_bin_interval=0.25)

from figure_scripts.heatmaps_grid_by_rating import plot_2heatmaps_grid_by_unique_ratings
plot_2heatmaps_grid_by_unique_ratings(train_samples.copy(), test_samples.copy(), MODEL_NAME, bin_interval=0.5)

from figure_scripts.ratings_preds_heatmaps_by_avgratings import plot_ratings_vs_preds_2dheatmaps_grid
plot_ratings_vs_preds_2dheatmaps_grid(train_samples.copy(), test_samples.copy(), MODEL_NAME, preds_bin_interval=0.5, avgs_bin_interval=.5)
plot_ratings_vs_preds_2dheatmaps_grid(train_samples.copy(), train_samples.copy(), MODEL_NAME, preds_bin_interval=0.5, avgs_bin_interval=.5)



