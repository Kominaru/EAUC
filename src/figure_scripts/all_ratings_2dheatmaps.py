import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def all_ratings_2dheatmap(train_samples : pd.DataFrame, test_samples: pd.DataFrame, MODEL_NAME: str, bin_interval: int = 0.5) -> None:

    # Calculate the average rating per user and movie in the train set
    user_avg_ratings = train_samples.groupby('user_id')['rating'].mean()
    movie_avg_ratings = train_samples.groupby('movie_id')['rating'].mean()
    
    train_samples['user_avg_rating'] = train_samples['user_id'].map(user_avg_ratings) 
    train_samples['movie_avg_rating'] = train_samples['movie_id'].map(movie_avg_ratings)

    test_samples['user_avg_rating'] = test_samples['user_id'].map(user_avg_ratings)
    test_samples['movie_avg_rating'] = test_samples['movie_id'].map(movie_avg_ratings)

    bins = np.arange(test_samples['rating'].min(), 5 + bin_interval, bin_interval)

    # Bin the average ratings into bins with the specified bin_interval
    train_samples['user_bin'] = pd.cut(train_samples['user_avg_rating'], bins=bins, include_lowest=True)
    train_samples['movie_bin'] = pd.cut(train_samples['movie_avg_rating'], bins=bins, include_lowest=True)

    test_samples['user_bin'] = pd.cut(test_samples['user_avg_rating'], bins=bins, include_lowest=True)
    test_samples['movie_bin'] = pd.cut(test_samples['movie_avg_rating'], bins=bins, include_lowest=True)

    # Calculate the (logarithmic) frequency for each combination of user_bin and movie_bin
    frequency_table = train_samples.groupby(['user_bin', 'movie_bin']).size().reset_index(name='frequency')
    frequency_table['log_frequency'] = frequency_table['frequency'].apply(lambda x: np.NaN if x == 0 else round(np.log10(x)))

    

    # Compute the RMSE in each (user average rating, movie average rating) bin
    test_samples['error'] = test_samples['rating'] - test_samples['pred']
    rmse_table = test_samples.groupby(['user_bin', 'movie_bin'])['error'].apply(lambda x: np.sqrt(np.mean(x**2))).reset_index(name='rmse')

    # Set to NaN the log_frequency and RMSE of the bins that have no samples in the train set
    frequency_table['log_frequency'] = frequency_table['log_frequency'].fillna(np.NaN)
    rmse_table['rmse'] = rmse_table['rmse'].fillna(np.NaN)
    
    # Create a pivot table to reshape the data for the heatmap
    pivot_table_freq = frequency_table.pivot(index='user_bin', columns='movie_bin', values='log_frequency')
    pivot_table_rmse = rmse_table.pivot(index='user_bin', columns='movie_bin', values='rmse')


    # Create a plot with the two heatmaps, one for the frequency and one for the RMSE

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    sns.heatmap(pivot_table_freq, cmap='Blues', ax=ax1, cbar_kws={'label': 'log10(frequency)'})
    ax1.set_xlabel('Average rating per movie')
    ax1.set_ylabel('Average rating per user')
    ax1.set_title('Heatmap of the Freq of the ratings\n per user and movie average ratings (train set)')
    # Change colorbar ticks to show the actual frequency in scientific notation
    cbar = ax1.collections[0].colorbar
    ticks = cbar.get_ticks()
    cbar.set_ticklabels(['$10^{' + str(int(tick)) + '}$' for tick in ticks])

    # Flip on the y-axis to have (0,0) in the bottom left corner
    ax1.invert_yaxis()

    # Reverse RdYlGn colormap to have green for the lowest RMSE and red for the highest RMSE
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    cmap = cmap.reversed()

    sns.heatmap(pivot_table_rmse, cmap=cmap, ax=ax2, cbar_kws={'label': 'RMSE'})
    ax2.set_xlabel('Average rating per movie')
    ax2.set_ylabel('Average rating per user')
    ax2.set_title('Heatmap of the RMSE of the predictions\n per user and movie average ratings (test set)')

    # Flip on the y-axis to have (0,0) in the bottom left corner
    ax2.invert_yaxis()

    plt.tight_layout()

    plt.savefig('figures/'+ MODEL_NAME + '/all_ratings_2dheatmap.pdf')
    plt.clf()