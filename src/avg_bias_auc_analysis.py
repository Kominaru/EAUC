# Script to compute, visualize and compare the AVG-BIAS-AUC for different models

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.avg_bias_auc import compute_avg_bias_auc

OUTPUT_DIR = "outputs/"


def plot_avg_bias_error(train_samples_dict: dict, test_samples_dict: dict = None, bins: int = 100) -> None:
    """
    Plots the error of the models depending on the distance from the real rating to the average of the user and item average ratings

    Parameters:
     - train_samples_dict: Dictionary with pairs (model_name: str, samples: pd.DataFrame)
     - test_samples_dict: Dictionary with pairs (model_name: str, samples: None | pd.DataFrame). The Train samples are used if no Test samples are provided
     - bins: Number of bins to cut the distance to the average rating

     *All DataFrames should have the following columns:
            - user_id: int
            - item_id: int
            - rating: float
            - pred: float
    """
    set_string = "Train" if test_samples_dict is None else "Test"
    test_samples_dict = test_samples_dict or train_samples_dict

    for model_name in train_samples_dict:
        train_samples = train_samples_dict[model_name]
        test_samples = test_samples_dict[model_name]

        # Map each sample to its user and item average rating
        test_samples["user_avg_rating"] = test_samples["user_id"].map(
            train_samples.groupby("user_id")["rating"].mean()
        )
        test_samples["item_avg_rating"] = test_samples["item_id"].map(
            train_samples.groupby("item_id")["rating"].mean()
        )

        # Compute the distance from the real rating to the average of the user and item average ratings
        test_samples["dist_to_avg_rating"] = (
            test_samples["rating"] - (test_samples["user_avg_rating"] + test_samples["item_avg_rating"]) / 2
        )

        # Compute the error of the model
        test_samples["error"] = (test_samples["rating"] - test_samples["pred"]).abs()

        # Cut into bins by the distance and compute the mean error for each bin
        errors_by_dist = test_samples.groupby(pd.cut(test_samples["dist_to_avg_rating"], bins))["error"].mean()

        errors_by_dist = errors_by_dist[~errors_by_dist.isna()]

        xx = errors_by_dist.index.to_series().apply(lambda x: x.mid)
        yy = errors_by_dist.values

        # Plot the error by distance
        plt.plot(xx, yy, label=model_name)

    # Plot the box that represents the maximum area of error. This box has a width of 2*(max_dist - min_dist) and a height of 2*(max_rating-min_rating)
    max_dist = test_samples["dist_to_avg_rating"].max()
    min_dist = test_samples["dist_to_avg_rating"].min()
    max_rating = test_samples["rating"].max()
    min_rating = test_samples["rating"].min()

    plt.plot(
        [min_dist, min_dist, max_dist, max_dist, min_dist],
        [0, (max_rating - min_rating), (max_rating - min_rating), 0, 0],
        label="Maximum area",
    )

    plt.xlabel("Distance to the average rating")
    plt.ylabel("MAE")
    plt.title(f"MAE by distance to the average rating ({set_string})")
    plt.legend()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=str, help="List of models to compare")
    parser.add_argument("--auc_method", type=str, default="ordered")
    parser.add_argument("--bins", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models = args.models
    auc_method = args.auc_method
    bins = args.bins

    # Load dataframes for each model
    model_predictions = {}
    for model in models:
        train_predictions = pd.read_csv(os.path.join(OUTPUT_DIR, model, "train_samples.csv"))
        test_predictions = pd.read_csv(os.path.join(OUTPUT_DIR, model, "test_samples_with_predictions.csv"))
        model_predictions[model] = {"train": train_predictions, "test": test_predictions}

    # Pretty print a table with the AVG-BIAS-AUC for each model
    print("AVG-BIAS-AUC")
    print("----------------------------------")
    print("Model               Train\tTest")
    for model in models:
        train_avg_bias_auc = compute_avg_bias_auc(model_predictions[model]["train"], method=auc_method, bins=bins)
        test_avg_bias_auc = compute_avg_bias_auc(
            model_predictions[model]["train"], model_predictions[model]["test"], method=auc_method, bins=bins
        )
        print(f"{model:<20}{train_avg_bias_auc:.4f}\t{test_avg_bias_auc:.4f}")
    print("------------")

    # Pretty print a table with the RMSE for each model
    print("RMSE")
    print("----------------------------------")
    print("Model               Train\tTest")
    for model in models:
        train_rmse = (
            (model_predictions[model]["train"]["rating"] - model_predictions[model]["train"]["pred"]) ** 2
        ).mean() ** 0.5
        test_rmse = (
            (model_predictions[model]["test"]["rating"] - model_predictions[model]["test"]["pred"]) ** 2
        ).mean() ** 0.5
        print(f"{model:<20}{train_rmse:.4f}\t{test_rmse:.4f}")
    print("------------")

    # Plot the train error by distance to the average rating for each model
    plot_avg_bias_error({model: model_predictions[model]["train"] for model in models}, bins=25)

    # Plot the test error by distance to the average rating for each model
    plot_avg_bias_error(
        {model: model_predictions[model]["train"] for model in models},
        {model: model_predictions[model]["test"] for model in models},
        bins=25,
    )
