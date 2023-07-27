import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_2dheatmap_ratings_vs_preds(test_samples: pd.DataFrame, model_name: str, preds_bin_interval = 0.25):

    preds_bins = np.arange(test_samples['rating'].min(), 5 + preds_bin_interval, preds_bin_interval)
    
    # The rating bins are the unique values of the ratings in the test set sorted in ascending order, adding 0 as the lower bound

    test_samples['pred_bin'] = pd.cut(test_samples['pred'], bins=preds_bins, include_lowest=True)

    # Calculate the logarithmic frequency for each combination of prediction and rating bin
    frequency_table = test_samples.groupby(['rating', 'pred_bin']).size().reset_index(name='frequency')
    frequency_table['log_frequency'] = frequency_table['frequency'].apply(lambda x: 0 if x == 0 else np.log10(x))

    # Create a pivot table to reshape the data for the heatmap
    pivot_table = frequency_table.pivot(index='rating', columns='pred_bin', values='log_frequency')

    # Create the heatmap
    plt.figure(figsize=(test_samples['rating'].nunique(), test_samples['rating'].nunique()))
    sns.heatmap(pivot_table, cmap='Blues', cbar_kws={'label': 'log10(frequency)'})
    plt.xlabel('Predicted Rating')
    plt.ylabel('Actual Rating')
    plt.title('Heatmap of the Freq of the ratings\n per actual and predicted ratings')

    # Flip on the y-axis to have (0,0) in the bottom left corner
    plt.gca().invert_yaxis()

    # Set colorbar ticks and labels
    colorbar = plt.gcf().axes[-1]
    colorbar.set_yticklabels([f'$10^{int(tick)}$' for tick in colorbar.get_yticks()])

    plt.tight_layout()
    plt.savefig('figures/'+ model_name + '/ratings_vs_preds_heatmap.pdf')
    plt.clf()