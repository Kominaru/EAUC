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
    "MF": "green",
    "MF_correction_lr": "red",
    "MF_correction_linear": "red",
    "Random": "gray",
    "GLOCAL_K": "blue",
    "GLOCAL_K_correction_lr": "lightblue",
    "GCMC": "red",
    "GCMC_correction_lr": "lightgreen",
    "BAYESIAN_SVD++": "violet",
    "BAYESIAN_SVD++_correction_lr": "burlywood",
    "MF_correction_linear_sigmoid": "orange",
}

CLEAN_NAMES = {
    "MF": "MF",
    "MF_correction_lr": "MF + Linear correction",
    "MF_correction_linear": "MF + Linear correction",
    "Random": "Random",
    "GLOCAL_K": "GLOCAL-K",
    "GCMC": "GC-MC",
    "BAYESIAN_SVD++": "Bayesian SVD++",
    "MF_correction_linear_sigmoid": "MF + Linear + Sigmoid correction",
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

    plt.figure(figsize=(5, 5))
    for model_name in train_samples_dict:
        xx = []
        yy = []
        for i in range(len(train_samples_dict[model_name])):
            train_samples = train_samples_dict[model_name][i]
            test_samples = test_samples_dict[model_name][i]

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
                (test_samples["rating"].max() - test_samples["rating"].min()),
                num_bins,
            )

            def group_rmse(df):
                return ((df["rating"] - df["pred"]) ** 2).mean() ** 0.5

            # Cut into bins by the distance and compute the RMSE in each bin
            # errors_by_dist = test_samples.groupby(pd.cut(test_samples["dist_to_avg_rating"], bins)).apply(group_rmse)
            # Compute the MAE in each bin
            errors_by_dist = test_samples.groupby(pd.cut(test_samples["dist_to_avg_rating"], bins))["error"].mean()

            errors_by_dist = errors_by_dist[~errors_by_dist.isna()]

            xx += errors_by_dist.index.to_series().apply(lambda x: x.mid).to_list()
            yy += errors_by_dist.values.tolist()

        df = pd.DataFrame({"xx": xx, "yy": yy})
        # Compute the mean and standard deviation of the error in each bin
        df = df.groupby("xx")["yy"].agg(["mean", "std"])
        df.dropna(inplace=True)

        xx = df.index
        yy = df["mean"]

        # Make axes square

        # Plot the error by distance
        plt.plot(xx, yy, label=CLEAN_NAMES[model_name], color=COLORS[model_name])

        # Plot the standard deviation of the error in each bin as a shaded area
        plt.fill_between(
            xx,
            yy - df["std"],
            yy + df["std"],
            alpha=0.1,
            color=COLORS[model_name],
        )

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

    # Plot y=x
    plt.plot([0, (max_rating - min_rating)], [0, (max_rating - min_rating)], color="black", alpha=0.2)

    # plt.plot(
    #     [min_dist, min_dist, max_dist, max_dist, min_dist],
    #     [0, (max_rating - min_rating), (max_rating - min_rating), 0, 0],
    #     label="Maximum area",
    # )

    plt.xlim(0, (max_rating - min_rating))
    plt.ylim(0, (max_rating - min_rating))
    plt.grid(alpha=0.3)

    plt.xlabel("Rating Eccentricity $Ecc_{ui}$")
    plt.ylabel("Prediction Error $|r_{ui} - \hat{r}_{ui}|$")
    # plt.title(f"{set_string.capitalize()} Prediction Error vs. Rating Eccentricity")
    plt.title("Netflix Prize")
    plt.legend(loc="upper left")

    plt.savefig(f"outputs/avg_bias_auc_{set_string.lower()}.pdf")
    plt.show()


def plot_ratings_vs_preds_lineplot(
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    dmr_range: tuple[int, int] = None,
) -> None:
    """
    Plots the ratings vs predictions of a model as a lineplot + shaded area chart

    Parameters:
     - train_samples: Train samples.
     - test_samples: Test samples.
     - dmr_range: Range of DMR_ui values to consider. If None, all samples are considered.
    """

    # Force a square figure
    plt.figure(figsize=(5, 5))

    # Compute the average rating per user and item
    user_avg_ratings = train_samples.groupby("user_id")["rating"].mean()
    item_avg_ratings = train_samples.groupby("item_id")["rating"].mean()

    test_samples["dmr_ui"] = (
        test_samples["user_id"].map(user_avg_ratings) + test_samples["item_id"].map(item_avg_ratings)
    ) / 2

    # Filter out samples for which the user or item average rating is not available
    test_samples = test_samples[~test_samples["dmr_ui"].isna()]

    # Filter out samples for which the user or item average rating is not in the specified range
    if dmr_range is not None:
        test_samples = test_samples[test_samples["dmr_ui"].between(*dmr_range)]

    # Plot the ratings vs predictions as a mean lineplot + shaded area chart

    rating_preds = test_samples.groupby("rating")["pred"].agg(["mean", "std"])

    plt.plot(rating_preds.index, rating_preds["mean"], color="red", label="Prediction Mean (MF)")
    plt.fill_between(
        rating_preds.index,
        rating_preds["mean"] - rating_preds["std"],
        rating_preds["mean"] + rating_preds["std"],
        alpha=0.05,
        color="red",
        label="Prediction Std (MF)",
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

    # plt.xlim(train_samples["rating"].min(), train_samples["rating"].max())
    # plt.ylim(train_samples["rating"].min(), train_samples["rating"].max())

    plt.xlim(1, 5)
    plt.ylim(1, 5)

    if dmr_range is not None:
        plt.title(
            f"Test ratings vs. predictions for $(u, i)$ \n pairs with ${dmr_range[0]} \leq DMR_{{ui}} \leq {dmr_range[1]}$"
        )
    else:
        plt.title(f"Test ratings vs. predictions\nfor all $(u, i)$ pairs")

    plt.legend()
    plt.tight_layout()

    plt.savefig(f"outputs/ratings_vs_preds_lineplot.pdf")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=str, help="List of models to compare")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--auc_method", type=str, default="ordered")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--tails", type=int, default=1)
    parser.add_argument("--exec_i", type=int)
    parser.add_argument("--exec_n", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models = args.models
    auc_method = args.auc_method
    bins = args.bins
    tails = args.tails
    dataset = args.dataset
    exec_i = args.exec_i
    exec_n = args.exec_n

    # Load dataframes for each model
    model_predictions = {}
    for model in models:
        if model not in ["Random", "average"]:
            train_predictions = []
            test_predictions = []
            for i in range(exec_i, exec_i + exec_n):
                train_predictions.append(
                    pd.read_csv(os.path.join(OUTPUT_DIR, dataset, model, f"train_outputs_{i}.csv"))
                )
                test_predictions.append(pd.read_csv(os.path.join(OUTPUT_DIR, dataset, model, f"test_outputs_{i}.csv")))

            model_predictions[model] = {"train": train_predictions, "test": test_predictions}

    plot_ratings_vs_preds_lineplot(
        model_predictions[models[0]]["train"][0], model_predictions[models[0]]["test"][0], dmr_range=None
    )

    plt.figure(figsize=(5, 5))

    # Print number of samples in test set for each model
    for model in models:
        print(f"{model}: {len(model_predictions[model]['test'][0]['pred'])}")
        print("Max prediction:", model_predictions[model]["test"][0]["pred"].max())
        print("Min prediction:", model_predictions[model]["test"][0]["pred"].min())
        print("NA predictions:", model_predictions[model]["test"][0]["pred"].isna().sum())

    for model in models:
        # Plot a histogram of the predictions of each model
        plt.hist(
            model_predictions[model]["test"][0]["pred"],
            bins=100,
            alpha=0.5,
            label=CLEAN_NAMES[model],
            color=COLORS[model],
        )

    plt.hist(model_predictions[model]["test"][0]["rating"], bins=100, alpha=0.5, label="Ratings", color="black")

    plt.xlabel("Prediction $\hat{r}_{ui}$")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predictions")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 5))
    for model in models:
        # Take the first test sample of each model and plot the distribution of errors
        plt.hist(
            (model_predictions[model]["test"][0]["rating"] - model_predictions[model]["test"][0]["pred"]).abs(),
            bins=100,
            alpha=0.5,
            label=CLEAN_NAMES[model],
            color=COLORS[model],
        )

    plt.xlabel("Prediction Absolute Error $|r_{ui} - \hat{r}_{ui}|$")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if "Random" in models:
        # Load the samples of a different model to compute the random predictions
        train_predictions = [
            model_predictions[models[0]]["train"][i].copy() for i in range(len(model_predictions[models[0]]["train"]))
        ]
        test_predictions = [
            model_predictions[models[0]]["test"][i].copy() for i in range(len(model_predictions[models[0]]["test"]))
        ]

        for i in range(len(train_predictions)):
            # Compute the random predictions
            train_predictions[i]["pred"] = np.random.uniform(
                train_predictions[i]["rating"].min(), train_predictions[i]["rating"].max(), len(train_predictions[i])
            )
            test_predictions[i]["pred"] = np.random.uniform(
                test_predictions[i]["rating"].min(), test_predictions[i]["rating"].max(), len(test_predictions[i])
            )

        model_predictions["Random"] = {"train": train_predictions, "test": test_predictions}

    if "average" in models:
        # Load the samples of a different model to compute the random predictions
        train_predictions = [
            model_predictions[models[0]]["train"][i].copy() for i in range(len(model_predictions[models[0]]["train"]))
        ]
        test_predictions = [
            model_predictions[models[0]]["test"][i].copy() for i in range(len(model_predictions[models[0]]["test"]))
        ]

        for i in range(len(train_predictions)):
            # A
            # Compute the average train ratings per user and item

            avg_user = train_predictions[i].groupby("user_id")["rating"].mean()
            avg_item = train_predictions[i].groupby("item_id")["rating"].mean()

            train_predictions[i]["pred"] = (
                train_predictions[i]["user_id"].map(avg_user) + train_predictions[i]["item_id"].map(avg_item)
            ) / 2

            test_predictions[i]["pred"] = (
                test_predictions[i]["user_id"].map(avg_user) + test_predictions[i]["item_id"].map(avg_item)
            ) / 2

        model_predictions["average"] = {"train": train_predictions, "test": test_predictions}

    # Pretty print a table with the RMSE for each model
    print("Prediction Error")
    print("----------------------------------")
    print("Model               Train\tTest")
    for model in models:
        for i in range(len(model_predictions[model]["train"])):
            train_rmse = (
                (model_predictions[model]["train"][i]["rating"] - model_predictions[model]["train"][i]["pred"]) ** 2
            ).mean() ** 0.5
            test_rmse = (
                (model_predictions[model]["test"][i]["rating"] - model_predictions[model]["test"][i]["pred"]) ** 2
            ).mean() ** 0.5
            print(f"{model:<20}{train_rmse:.3f}\t{test_rmse:.3f}")
    print("------------")

    print("MAE")
    print("----------------------------------")
    print("Model               Train\tTest")
    for model in models:
        for i in range(len(model_predictions[model]["train"])):
            train_mae = (
                (model_predictions[model]["train"][i]["rating"] - model_predictions[model]["train"][i]["pred"])
                .abs()
                .mean()
            )
            test_mae = (
                (model_predictions[model]["test"][i]["rating"] - model_predictions[model]["test"][i]["pred"])
                .abs()
                .mean()
            )
            print(f"{model:<20}{train_mae:.3f}\t{test_mae:.3f}")

    # Pretty print a table with the AVG-BIAS-AUC for each model
    print("AVG-BIAS-AUC")
    print("----------------------------------")
    print("Model               Train\tTest")
    for model in models:
        for i in range(len(model_predictions[model]["train"])):
            train_avg_bias_auc = compute_avg_bias_auc(
                model_predictions[model]["train"][i], method=auc_method, tails=tails, bins=bins
            )
            test_avg_bias_auc = compute_avg_bias_auc(
                model_predictions[model]["train"][i],
                model_predictions[model]["test"][i],
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
