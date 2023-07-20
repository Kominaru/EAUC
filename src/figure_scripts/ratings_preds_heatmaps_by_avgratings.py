import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_ratings_vs_preds_2dheatmaps_grid(train_samples: pd.DataFrame, test_samples: pd.DataFrame, model_name:str, preds_bin_interval: int = 0.5, avgs_bin_interval: int = 0.5) -> None:

    ## Plot a 2d grid of heatmaps of predicted vs actual rating for each bin of user and movie average ratings, making the 
    ## color levels depend on the overall frequency of ratings rather than the frequency of ratings in each bin

    # Calculate the average rating per user and movie in the train set

    preds_bins = np.arange(test_samples['rating'].min(), 5 + preds_bin_interval, preds_bin_interval)
    avg_ratings_bins = np.arange(test_samples['rating'].min(), 5 + avgs_bin_interval, avgs_bin_interval)

    user_avg_ratings = train_samples.groupby('user_id')['rating'].mean()
    movie_avg_ratings = train_samples.groupby('movie_id')['rating'].mean()

    test_samples['user_avg_rating'] = test_samples['user_id'].map(user_avg_ratings)
    test_samples['movie_avg_rating'] = test_samples['movie_id'].map(movie_avg_ratings)

    test_samples['user_bin'] = pd.cut(test_samples['user_avg_rating'], bins=avg_ratings_bins, include_lowest=True)
    test_samples['movie_bin'] = pd.cut(test_samples['movie_avg_rating'], bins=avg_ratings_bins, include_lowest=True)

    user_bins = list(test_samples['user_bin'].unique().categories)

    

    # Precompute the logarithmic frequency for all combinations of rating and prediction
    test_samples['prediction_bin'] = pd.cut(test_samples['prediction'], bins=preds_bins, include_lowest=True)

    frequency_table = test_samples.groupby(['user_bin', 'movie_bin', 'rating', 'prediction_bin']).size().reset_index(name='frequency')
    frequency_table['log_frequency'] = frequency_table['frequency'].apply(lambda x: np.NaN if x == 0 else np.log10(x))

    # Create a grid of plots
    fig, axs = plt.subplots(len(user_bins), len(user_bins), figsize=(5*(len(user_bins)), 5*(len(user_bins))))

    # For each combination of user_bin and movie_bin, create a heatmap of the frequency of ratings vs the prediction
    for i, user_bin in enumerate(user_bins):
        for j, movie_bin in enumerate(user_bins):
            df = frequency_table[(frequency_table['user_bin'] == user_bin) & (frequency_table['movie_bin'] == movie_bin)]
            
            # If all frequencies are NaN, skip this subplot making it fully white
            if df['log_frequency'].isna().all():
                axs[i, j].axis('off')
                continue



            # Create the pivot table to reshape the data for the heatmap
            pivot_table = df.pivot(index='prediction_bin', columns='rating', values='log_frequency')

            # Create the heatmap
            sns.heatmap(pivot_table, cmap='Blues', cbar_kws={'label': 'log10(frequency)'}, ax=axs[i, j], vmin=0, vmax=frequency_table['log_frequency'].max())
            axs[i, j].set_xlabel('Actual Rating')
            axs[i, j].set_ylabel('Predicted Rating')
            axs[i, j].set_title(f'user avg {user_bin}\n movie avg {movie_bin}')

            # Make sure the heatmap is square by making each subplot have square axes
            axs[i, j].set_box_aspect(1)

            # Flip on the y-axis to have (0,0) in the bottom left corner
            axs[i, j].invert_yaxis()

            # Remove colorbar
            axs[i, j].collections[0].colorbar.remove()

            # Plot an x=y line
            axs[i, j].plot([0, test_samples['rating'].nunique()], [0, test_samples['prediction_bin'].nunique()], color='black', linewidth=1)

    plt.tight_layout()
    plt.savefig('figures/' + model_name + '/ratings_vs_predictions_2d_heatmaps_grid.pdf')