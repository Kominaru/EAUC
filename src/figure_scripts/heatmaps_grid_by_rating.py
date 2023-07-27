import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_2heatmaps_grid_by_unique_ratings(train_samples: pd.DataFrame, test_samples: pd.DataFrame, MODEL_NAME: str, bin_interval: float = 0.5) -> None:

    # Use half ratings (0.5, 1.5, 2.5, ...) or full ratings (1, 2, 3, ...)?
    # Just need to look at the test set to know if we need to use half ratings or not
    if test_samples['rating'].min() == 0.5:
        ratings = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    else:
        ratings = [1, 2, 3, 4, 5]

    bins = np.arange(1, 5 + bin_interval, bin_interval)

    fig, axs = plt.subplots(len(ratings), 3, figsize=(5*3, 5*len(ratings)))

    # Calculate the average rating per user and movie in the train set
    user_avg_ratings = train_samples.groupby('user_id')['rating'].mean()
    movie_avg_ratings = train_samples.groupby('movie_id')['rating'].mean()

    test_samples['user_avg_rating'] = test_samples['user_id'].map(user_avg_ratings).reset_index(drop=True)
    test_samples['movie_avg_rating'] = test_samples['movie_id'].map(movie_avg_ratings).reset_index(drop=True)

    test_samples['user_bin'] = pd.cut(test_samples['user_avg_rating'], bins=bins, include_lowest=True)
    test_samples['movie_bin'] = pd.cut(test_samples['movie_avg_rating'], bins=bins, include_lowest=True)

    train_samples['user_avg_rating'] = train_samples['user_id'].map(user_avg_ratings).reset_index(drop=True)
    train_samples['movie_avg_rating'] = train_samples['movie_id'].map(movie_avg_ratings).reset_index(drop=True)

    train_samples['user_bin'] = pd.cut(train_samples['user_avg_rating'], bins=bins, include_lowest=True)
    train_samples['movie_bin'] = pd.cut(train_samples['movie_avg_rating'], bins=bins, include_lowest=True)

    test_samples['error'] = test_samples['rating'] - test_samples['pred']

    for i, rating in enumerate(ratings):

        cmap = plt.get_cmap('RdYlGn').reversed()

        # Create the plot for the RMSE. Green means low RMSE (good), Red means high RMSE (bad)
        rmse_table = test_samples[test_samples['rating'] == rating].groupby(['user_bin', 'movie_bin'])['error'].apply(lambda x: np.sqrt(np.mean(x**2))).reset_index(name='rmse')
        pivot_table = rmse_table.pivot(index='user_bin', columns='movie_bin', values='rmse')
        sns.heatmap(pivot_table.iloc[::-1], cmap=cmap, annot=False, cbar_kws={'label': 'RMSE'}, fmt='g', ax=axs[i, 0], vmin=0, vmax=2)
        axs[i, 0].set_ylabel('User\'s Average Rating')
        axs[i, 0].set_title(f'RMSE for rating {rating}')

        # Create the plot for the frequency
        frequency_table = test_samples[test_samples['rating'] == rating].groupby(['user_bin', 'movie_bin']).size().reset_index(name='frequency')
        frequency_table['log_frequency'] = frequency_table['frequency'].apply(lambda x: np.NaN if x == 0 else round(np.log10(x)))
        pivot_table = frequency_table.pivot(index='user_bin', columns='movie_bin', values='log_frequency')
        sns.heatmap(pivot_table.iloc[::-1], cmap='YlGnBu', annot=False, cbar_kws={'label': 'Logarithmic Frequency'}, fmt='g', ax=axs[i, 1], vmin=0)
        axs[i, 1].set_xlabel('Movie\'s Average Rating')
        axs[i, 1].set_ylabel('User\'s Average Rating')
        axs[i, 1].set_title(f'Frequency for rating {rating}')

        # Change the colorbar ticks and labels
        colorbar = axs[i, 1].collections[0].colorbar
        colorbar.set_ticks([0, 1, 2, 3, 4, 5])
        colorbar.set_ticklabels([f'$10^{int(tick)}$' for tick in colorbar.get_ticks()])
        colorbar.set_label('Frequency (log10)')

        # Create the plot for the train set frequency
        frequency_table = train_samples[train_samples['rating'] == rating].groupby(['user_bin', 'movie_bin']).size().reset_index(name='frequency')
        frequency_table['log_frequency'] = frequency_table['frequency'].apply(lambda x: np.NaN if x == 0 else round(np.log10(x)))
        pivot_table = frequency_table.pivot(index='user_bin', columns='movie_bin', values='log_frequency')
        sns.heatmap(pivot_table.iloc[::-1], cmap='YlGnBu', annot=False, cbar_kws={'label': 'Logarithmic Frequency'}, fmt='g', ax=axs[i, 2], vmin=0)
        axs[i, 2].set_xlabel('Movie\'s Average Rating')
        axs[i, 2].set_ylabel('User\'s Average Rating')
        axs[i, 2].set_title(f'Frequency for rating {rating} (train set)')

        # Change the colorbar ticks and labels
        colorbar = axs[i, 2].collections[0].colorbar
        colorbar.set_ticks([0, 1, 2, 3, 4, 5])
        colorbar.set_ticklabels([f'$10^{int(tick)}$' for tick in colorbar.get_ticks()])
        colorbar.set_label('Frequency (log10)')


        


    plt.tight_layout()
    plt.savefig('figures/'+ MODEL_NAME + '/rmse_and_frequency_heatmaps.pdf')
    plt.clf()
