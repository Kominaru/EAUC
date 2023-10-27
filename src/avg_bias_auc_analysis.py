# Script to compute, visualize and compare the AVG-BIAS-AUC for different models

import os
import sys
import argparse
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.avg_bias_auc import compute_avg_bias_auc

OUTPUT_DIR = "outputs/"

COLORS = {
    "MF": "red",
    "MF_correction_lr": "lightcoral",
    "random": "gray",
    "GLOCAL_K": "blue",
    "GLOCAL_K_correction_lr": "lightblue",
    "GC_MC": "green",
    "GC_MC_correction_lr": "lightgreen",
    "BAYESIAN_SVD++": "brown",
    "BAYESIAN_SVD++_correction_lr": "burlywood",
}


def plot_avg_bias_error(
    train_samples_dict: dict, test_samples_dict: dict = None, num_bins: int = 100, tails: Literal[1, 2] = 1
) -> None:
    """
    Plots the error of the models depending on the distance from the real rating to the average of the user and item average ratings

    Parameters:
     - train_samples_dict: Dictionary with pairs (model_name: str, samples: pd.DataFrame)
     - test_samples_dict: Dictionary with pairs (model_name: str, samples: None | pd.DataFrame). The Train samples are used if no Test samples are provided
     - num_bins: Number of bins to cut the distance to the average rating
     - tails: Whether to consider a double-tailed or single-tailed AVG-BIAS-AUC (i.e., whether to consider the absolute value of the distance or not)

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

        if tails == 1:
            test_samples["dist_to_avg_rating"] = test_samples["dist_to_avg_rating"].abs()

        # Compute the error of the model
        test_samples["error"] = (test_samples["rating"] - test_samples["pred"]).abs()

        left = -(test_samples["rating"].max() - test_samples["rating"].min()) if tails == 2 else 0

        bins = np.linspace(
            left,
            +(test_samples["rating"].max() - test_samples["rating"].min()),
            num_bins,
        )

        def group_rmse(df):
            return ((df["rating"] - df["pred"]) ** 2).mean() ** 0.5

        # Cut into bins by the distance and compute the RMSE in each bin
        # errors_by_dist = test_samples.groupby(pd.cut(test_samples["dist_to_avg_rating"], bins)).apply(group_rmse)
        # Compute the MAE in each bin
        errors_by_dist = test_samples.groupby(pd.cut(test_samples["dist_to_avg_rating"], bins))["error"].mean()

        errors_by_dist = errors_by_dist[~errors_by_dist.isna()]

        xx = errors_by_dist.index.to_series().apply(lambda x: x.mid)
        yy = errors_by_dist.values

        # Plot the error by distance
        plt.plot(xx, yy, label=model_name, color=COLORS[model_name])

        # Plot the RMSE of the model on the whole set as a horizontal line
        # plt.plot(
        #     [xx.min(), xx.max()],
        #     [group_rmse(test_samples), group_rmse(test_samples)],
        #     label=f"{model_name} (overall RMSE)",
        #     color=COLORS[model_name],
        #     linestyle="--",
        #     alpha=0.5,
        # )

    # Plot the box that represents the maximum area of error. This box has a width of 2*(max_dist - min_dist) and a height of 2*(max_rating-min_rating)
    max_dist = test_samples["dist_to_avg_rating"].max()
    min_dist = test_samples["dist_to_avg_rating"].min() if tails == 2 else 0
    max_rating = test_samples["rating"].max()
    min_rating = test_samples["rating"].min()

    # plt.plot(
    #     [min_dist, min_dist, max_dist, max_dist, min_dist],
    #     [0, (max_rating - min_rating), (max_rating - min_rating), 0, 0],
    #     label="Maximum area",
    # )

    plt.xlim(min_dist, max_dist)
    plt.ylim(0, (max_rating - min_rating))
    plt.grid(alpha=0.3)

    plt.xlabel("Rating eccentricity $Ecc_{ui}$")
    plt.ylabel("RMSE")
    plt.title(f"{set_string.capitalize()} RMSE by rating eccentricity")
    plt.legend()

    plt.savefig(f"outputs/avg_bias_auc_{set_string.lower()}.pdf")
    plt.show()


def plot_ratings_vs_preds_lineplot(
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    user_avg_rating_range: tuple[int, int] = None,
    item_avg_rating_range: tuple[int, int] = None,
) -> None:
    """
    Plots the ratings vs predictions of a model as a lineplot + shaded area chart

    Parameters:
     - train_samples: Train samples.
     - test_samples: Test samples.
     - user_avg_rating_range: Range of user average ratings to consider (min, max). If None, the whole range is considered.
     - item_avg_rating_range: Range of item average ratings to consider (min, max). If None, the whole range is considered.
    """

    # Force a square figure
    plt.figure(figsize=(5, 5))

    # Compute the average rating per user and item
    user_avg_ratings = train_samples.groupby("user_id")["rating"].mean()
    item_avg_ratings = train_samples.groupby("item_id")["rating"].mean()

    test_samples["user_avg_rating"] = test_samples["user_id"].map(user_avg_ratings)
    test_samples["item_avg_rating"] = test_samples["item_id"].map(item_avg_ratings)

    # Filter out samples for which the user or item average rating is not available
    test_samples = test_samples[~test_samples["user_avg_rating"].isna()]
    test_samples = test_samples[~test_samples["item_avg_rating"].isna()]

    # Filter out samples for which the user or item average rating is not in the specified range
    if user_avg_rating_range is not None:
        test_samples = test_samples[test_samples["user_avg_rating"].between(*user_avg_rating_range)]

    if item_avg_rating_range is not None:
        test_samples = test_samples[test_samples["item_avg_rating"].between(*item_avg_rating_range)]

    # Plot the ratings vs predictions as a mean lineplot + shaded area chart

    rating_preds = test_samples.groupby("rating")["pred"].agg(["mean", "std"])

    plt.plot(rating_preds.index, rating_preds["mean"], color="red", label="Mean prediction")
    plt.fill_between(
        rating_preds.index,
        rating_preds["mean"] - rating_preds["std"],
        rating_preds["mean"] + rating_preds["std"],
        alpha=0.05,
        color="red",
        label="Standard deviation",
    )

    # Plot the ideal line
    plt.plot(
        [test_samples["rating"].min(), test_samples["rating"].max()],
        [test_samples["rating"].min(), test_samples["rating"].max()],
        color="black",
        label="Ideal regressor",
        linestyle="--",
    )

    plt.xlabel("Rating $r_{ui}$")
    plt.ylabel("Prediction $\hat{r}_{ui}$")

    plt.xlim(train_samples["rating"].min(), train_samples["rating"].max())
    plt.ylim(train_samples["rating"].min(), train_samples["rating"].max())

    plt.title(
        f"Test Ratings vs predictions for users and items \n with average train ratings in {user_avg_rating_range} and {item_avg_rating_range}"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"outputs/ratings_vs_preds_lineplot.pdf")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=str, help="List of models to compare")
    parser.add_argument("--auc_method", type=str, default="ordered")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--tails", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models = args.models
    auc_method = args.auc_method
    bins = args.bins
    tails = args.tails

    # Load dataframes for each model
    model_predictions = {}
    for model in models:
        if model != "random":
            train_predictions = pd.read_csv(os.path.join(OUTPUT_DIR, model, "train_samples.csv"))
            test_predictions = pd.read_csv(os.path.join(OUTPUT_DIR, model, "test_samples_with_predictions.csv"))
            model_predictions[model] = {"train": train_predictions, "test": test_predictions}
            # plot_ratings_vs_preds_lineplot(
            #     train_predictions, test_predictions, user_avg_rating_range=(4.5, 5.0), item_avg_rating_range=(4.5, 5.0)
            # )

    if "random" in models:
        # Load the samples of a different model to compute the random predictions
        train_predictions = model_predictions[models[0]]["train"].copy()
        test_predictions = model_predictions[models[0]]["test"].copy()

        # Compute the random predictions
        train_predictions["pred"] = np.random.uniform(
            train_predictions["rating"].min(), train_predictions["rating"].max(), len(train_predictions)
        )
        test_predictions["pred"] = np.random.uniform(
            test_predictions["rating"].min(), test_predictions["rating"].max(), len(test_predictions)
        )
        model_predictions["random"] = {"train": train_predictions, "test": test_predictions}

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
        print(f"{model:<20}{train_rmse:.3f}\t{test_rmse:.3f}")
    print("------------")

    # Pretty print a table with the AVG-BIAS-AUC for each model
    print("AVG-BIAS-AUC")
    print("----------------------------------")
    print("Model               Train\tTest")
    for model in models:
        train_avg_bias_auc = compute_avg_bias_auc(
            model_predictions[model]["train"], method=auc_method, tails=tails, bins=bins
        )
        test_avg_bias_auc = compute_avg_bias_auc(
            model_predictions[model]["train"],
            model_predictions[model]["test"],
            method=auc_method,
            tails=tails,
            bins=bins,
        )
        print(f"{model:<20}{train_avg_bias_auc:.3f}\t{test_avg_bias_auc:.3f}")
    print("------------")

    # Plot the train error by distance to the average rating for each model
    plot_avg_bias_error({model: model_predictions[model]["train"] for model in models}, num_bins=25, tails=tails)

    # Plot the test error by distance to the average rating for each model
    plot_avg_bias_error(
        {model: model_predictions[model]["train"] for model in models},
        {model: model_predictions[model]["test"] for model in models},
        num_bins=25,
        tails=tails,
    )
