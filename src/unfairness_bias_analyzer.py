# Path: unfairness_bias_analyzer.py
import pandas as pd
import numpy as np

from figure_scripts.basic_dataset_statistics import plot_dataset_statistics_figures
from figure_scripts.heatmaps_avgrating import plot_ratings_vs_preds_2dheatmaps_grid
from figure_scripts.all_ratings_2dheatmaps import all_ratings_2dheatmap
from figure_scripts.rating_preds_conf_matrix import plot_2dheatmap_ratings_vs_preds
from figure_scripts.heatmaps_grid_by_rating import plot_2heatmaps_grid_by_unique_ratings
import os
import warnings

# Supress FixedFormatter FixedLocator warnings
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "GLOCAL_K_correction_lr"

os.makedirs(f"figures/{MODEL_NAME}/", exist_ok=True)


def print_basic_dataset_statistics(train_samples: pd.DataFrame, test_samples: pd.DataFrame):
    """
    1) Prints basic statistics about the dataset:
    - Number of ratings, users and items
    - Min and max ratings and predictions
    - Number of users in the test set but not in the training set
    - Number of train samples repeated in the test set

    2) Plots the basic dataset statistics figures:
    - Histogram of the ratings as a whole with logarithmic y-scale
    - Histogram of the average rating per item (train set)
    - Histogram of the average rating per user (train set)

    Parameters:
        train_samples (pd.DataFrame): Train samples.
        test_samples (pd.DataFrame): Test samples.
    """

    all_samples = pd.concat([train_samples, test_samples])
    print("=====================================")
    print("Basic dataset statistics:")
    print("=====================================")

    print(f"    #Ratings:\t\t\t\t{len(all_samples)}")
    print(f'    #Users:\t\t\t\t{len(all_samples["user_id"].unique())}')
    print(f'    #Items:\t\t\t\t{len(all_samples["item_id"].unique())}')

    # Min and max ratings
    min_rating, max_rating = all_samples["rating"].min(), all_samples["rating"].max()
    min_pred, max_pred = all_samples["pred"].min(), all_samples["pred"].max()

    print(f"    Min-max ratings:\t\t\t{min_rating:.1f} - {max_rating:.1f}")
    print(f"    Min-max predictions:\t\t{min_pred:.1f} - {max_pred:.1f}")

    # How many users are in the test set but not in the training set?
    print(f'    #Unseen test users:\t\t\t{len(test_samples[~test_samples["user_id"].isin(train_samples["user_id"])])}')

    train_tuples = train_samples[["user_id", "item_id"]].apply(tuple, axis=1)
    test_tuples = test_samples[["user_id", "item_id"]].apply(tuple, axis=1)

    print(f"    #Repeated train-test samples:\t{test_tuples.isin(train_tuples).sum()}")

    # Plot figures
    plot_dataset_statistics_figures(all_samples.copy(), train_samples.copy(), MODEL_NAME)


def compute_rmse(train_samples: pd.DataFrame, test_samples: pd.DataFrame):
    """
    Computes the RMSE on the train and test sets.

    Parameters:
        train_samples (pd.DataFrame): Train samples.
        test_samples (pd.DataFrame): Test samples.

    """
    print("=====================================")
    print("RMSE")
    print("=====================================")

    # Compute the RMSE on the train set
    rmse = ((train_samples["rating"] - train_samples["pred"]) ** 2).mean() ** 0.5
    print(f"    Train: {rmse:.3f}")

    # Compute the RMSE by rating on the train set
    for rating in train_samples["rating"].unique():
        rmse = (
            (
                train_samples[train_samples["rating"] == rating]["rating"]
                - train_samples[train_samples["rating"] == rating]["pred"]
            )
            ** 2
        ).mean() ** 0.5
        print(f"        Rating {rating}: {rmse:.3f}")

    # Compute the RMSE on the test set
    rmse = ((test_samples["rating"] - test_samples["pred"]) ** 2).mean() ** 0.5
    print(f"    Test:  {rmse:.3f}")

    # Compute the RMSE by rating on the test set
    for rating in test_samples["rating"].unique():
        rmse = (
            (
                test_samples[test_samples["rating"] == rating]["rating"]
                - test_samples[test_samples["rating"] == rating]["pred"]
            )
            ** 2
        ).mean() ** 0.5
        print(f"        Rating {rating}: {rmse:.3f}")

    # Compute the RMSE (global and by rating) using a random baseline
    random_preds = np.random.uniform(low=1.0, high=5.0, size=len(test_samples))
    rmse = ((test_samples["rating"] - random_preds) ** 2).mean() ** 0.5
    print(f"    Random baseline: {rmse:.3f}")

    for rating in test_samples["rating"].unique():
        rmse = (
            (test_samples[test_samples["rating"] == rating]["rating"] - random_preds[test_samples["rating"] == rating])
            ** 2
        ).mean() ** 0.5
        print(f"        Rating {rating}: {rmse:.3f}")


def plot_model_prediction_analysis(train_samples, test_samples):
    """
    Plots the figures for the model prediction analysis:
    1. Heatmap of the frequency of the ratings depending on user and item average ratings (train set)
    2. Heatmap of the frequency of test samples depending on rating and prediction bins (test set)
    3. Heatmap of the RMSE depending on user and item average ratings (train set) separated by unique ratings
    4. Heatmap of the frequency of test samples depending on rating and prediction bins (test set)
       separated by the user and item average ratings (train set)

    Parameters:
        train_samples (pd.DataFrame): Train samples.
        test_samples (pd.DataFrame): Test samples.
    """

    print("=====================================")
    print("Model prediction bias analysis plots")
    print("=====================================")
    print("1) Heatmap of the frequency of the ratings depending on user and item average ratings (train set)")
    all_ratings_2dheatmap(train_samples.copy(), test_samples.copy(), MODEL_NAME, bin_interval=0.5)

    print("2) Heatmap of the frequency of test samples depending on rating and prediction bins (test set)")
    plot_2dheatmap_ratings_vs_preds(test_samples.copy(), MODEL_NAME, preds_bin_interval=0.25)

    print("3) Heatmap of the RMSE depending on user and item average ratings (train set) separated by unique ratings")
    plot_2heatmaps_grid_by_unique_ratings(train_samples.copy(), test_samples.copy(), MODEL_NAME, bin_interval=0.5)

    print(
        "4) Heatmap of the frequency of test samples depending on rating and prediction bins (test set) separated by the user and item average ratings (train set)"
    )
    plot_ratings_vs_preds_2dheatmaps_grid(
        train_samples.copy(),
        test_samples.copy(),
        MODEL_NAME,
        preds_bin_interval=0.5,
        avgs_bin_interval=0.5,
    )

    plot_ratings_vs_preds_2dheatmaps_grid(
        train_samples.copy(),
        train_samples.copy(),
        MODEL_NAME,
        preds_bin_interval=0.5,
        avgs_bin_interval=0.5,
    )


if __name__ == "__main__":
    # Load train and test samples
    train_samples: pd.DataFrame = pd.read_csv(f"outputs/{MODEL_NAME}/train_samples.csv")
    test_samples: pd.DataFrame = pd.read_csv(f"outputs/{MODEL_NAME}/test_samples_with_predictions.csv")

    # Print basic dataset statistics
    print_basic_dataset_statistics(train_samples, test_samples)

    # Clamp predictions to [1.0, 5.0]
    test_samples["pred"] = np.clip(test_samples["pred"], 1.0, 5.0)
    train_samples["pred"] = np.clip(train_samples["pred"], 1.0, 5.0)

    # Brief check of the RMSE to ensure we're reproducing the results from the paper

    compute_rmse(train_samples, test_samples)

    # Create directory for figures
    os.makedirs("figures/" + MODEL_NAME + "/", exist_ok=True)

    # Plot figures for the model predictions bias analysis
    plot_model_prediction_analysis(train_samples, test_samples)
