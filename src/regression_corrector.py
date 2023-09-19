import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

MODEL_NAME = "MF"

train_samples: pd.DataFrame = pd.read_csv(f"outputs/{MODEL_NAME}/train_samples.csv")
test_samples: pd.DataFrame = pd.read_csv(f"outputs/{MODEL_NAME}/test_samples_with_predictions.csv")

with pd.concat([train_samples, test_samples]) as all_samples:
    print(f"==============================")
    print(f"Dataset statistics")
    print(f"==============================")

    print(f"Number of ratings: {len(all_samples)}")
    print(f"Number of users: {all_samples['user_id'].nunique()}")
    print(f"Number of movies: {all_samples['movie_id'].nunique()}")

# Compute the average rating per user and per movie
user_avg_ratings = train_samples.groupby("user_id")["rating"].mean()
movie_avg_ratings = train_samples.groupby("movie_id")["rating"].mean()

# Add the avg. ratings as a column to the train and test samples
train_samples["user_avg_rating"] = train_samples["user_id"].map(user_avg_ratings)
train_samples["movie_avg_rating"] = train_samples["movie_id"].map(movie_avg_ratings)
test_samples["user_avg_rating"] = test_samples["user_id"].map(user_avg_ratings)
test_samples["movie_avg_rating"] = test_samples["movie_id"].map(movie_avg_ratings)

# Filter the train and test samples by the average rating of the user and the movie
USER_AVG_RATING_RANGE = (0, 5)
MOVIE_AVG_RATING_RANGE = (0, 5)

train_samples = train_samples[
    train_samples["user_avg_rating"].between(*USER_AVG_RATING_RANGE)
    & train_samples["movie_avg_rating"].between(*MOVIE_AVG_RATING_RANGE)
]
test_samples = test_samples[
    test_samples["user_avg_rating"].between(*USER_AVG_RATING_RANGE)
    & test_samples["movie_avg_rating"].between(*MOVIE_AVG_RATING_RANGE)
]

#############################
# Linear regression corrector
#############################

lr = LinearRegression()
lr.fit(train_samples[["rating", "user_avg_rating", "movie_avg_rating"]], train_samples["pred"])

print(
    f"""Linear equation: 
            pred = {lr.intercept_:.3f} + {lr.coef_[0]:.3f} * rating + {lr.coef_[1]:.3f} * user_avg_rating + {lr.coef_[2]:.3f} * movie_avg_rating"""
)


def correct_prediction(sample, lr):
    """
    Return the corrected prediction of a sample using the linear regression by solving the equation for rating.
    Original prediction: pred = a + b * rating + c * user_avg_rating + d * movie_avg_rating
    Solveing rating: rating = (pred - a - c * user_avg_rating - d * movie_avg_rating) / b

    Params:
        sample: A Pandas Row with the columns "rating", "user_avg_rating", "movie_avg_rating" and "pred"
        lr: the linear regression model

    Returns:
        The corrected prediction (predicted rating) clipped to the range [1, 5]
    """

    corrected_pred = (
        (
            sample["pred"]
            - lr.intercept_
            - lr.coef_[1] * sample["user_avg_rating"]
            - lr.coef_[2] * sample["movie_avg_rating"]
        )
        / lr.coef_[0]
    ).clip(1, 5)

    return corrected_pred


train_samples["pred"] = train_samples.apply(correct_prediction, args=[lr], axis=1)
test_samples["pred"] = test_samples.apply(correct_prediction, args=[lr], axis=1)

train_samples = train_samples[["user_id", "movie_id", "rating", "pred"]]
test_samples = test_samples[["user_id", "movie_id", "rating", "pred"]]

os.makedirs(f"outputs/{MODEL_NAME}_correction_lr", exist_ok=True)

train_samples.to_csv(f"outputs/{MODEL_NAME}_correction_lr/train_samples.csv", index=False)
test_samples.to_csv(f"outputs/{MODEL_NAME}_correction_lr/test_samples_with_predictions.csv", index=False)

exit()


def plot_error_distribution_by_difference_rating_to_avg_rating(samples, predicts_dict, bins=20):
    samples = samples.copy()

    samples["dist_avg_to_rating"] = (samples["user_avg_rating"] + samples["movie_avg_rating"]) / 2 - samples["rating"]

    samples["dist_avg_to_rating_bin"] = pd.cut(samples["dist_avg_to_rating"], bins=np.linspace(-4, 4, bins + 1))

    models_errors_dists = {}
    models_errors_bins = {}

    plt.figure(figsize=(7, 7))

    # Check the stability of the AUC metric depending on the number of bins

    plt.xscale("log")
    plt.xlabel("Number of bins")
    plt.ylabel("AUC")
    plt.legend()
    plt.show()

    for model_name, predicts in predicts_dict.items():
        samples["model_error"] = abs(samples["rating"] - predicts)

        samples_tmp = samples.groupby("dist_avg_to_rating_bin").filter(lambda x: len(x) >= 10).reset_index(drop=True)

        ################
        # 1) Data for Plot A

        models_errors_bins[model_name] = []

        models_errors_bins[model_name] = (
            samples_tmp.groupby("dist_avg_to_rating_bin")["model_error"]  # Compute the error for each bin
            .agg(["mean", "std"])
            .reset_index()
        )

        ################
        # 2) For Plot B

        models_errors_dists[model_name] = []

        for min_dist in np.arange(0, 4.5, 0.1):  # Compute the error for ratings with distance to avg >= min_dist
            filtered_samples = samples_tmp[samples_tmp["dist_avg_to_rating"] >= min_dist]

            mae = filtered_samples["model_error"].mean()
            std = filtered_samples["model_error"].std()

            models_errors_dists[model_name].append((mae, std))

    ###############
    # 1) Plot A)

    for model_name, model_errors in models_errors_bins.items():
        # Filter NaNs
        model_errors = model_errors.dropna()
        xx = model_errors["dist_avg_to_rating_bin"].apply(lambda x: x.mid)

        plt.plot(xx, model_errors["mean"], linewidth=2, label=model_name)  # Plot MAE per bin
        plt.fill_between(  # Fill std AE area for each bin
            xx,
            model_errors["mean"] - model_errors["std"],
            model_errors["mean"] + model_errors["std"],
            alpha=0.2,
        )

    plt.plot([-5, 5], [0, 0])  # Perfect regressor

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xticks(np.arange(-5, 5, 1))
    plt.yticks(np.arange(-5, 5, 1))
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.grid(which="major", axis="both", linestyle="-", linewidth=0.5)
    plt.grid(which="minor", axis="both", linestyle=":", linewidth=0.5)

    plt.xlabel("Avg(avgrating(user),avgrating(movie)) - rating")
    plt.ylabel("Prediction error")

    plt.legend()
    plt.tight_layout()
    plt.show()

    ################
    # Plot B)
    # Plot the MAE for each model as a function of the min_dist
    plt.figure(figsize=(7, 7))

    for model_name, errors in models_errors_dists.items():
        # Plot MAE and std error
        plt.plot(
            np.arange(0, 4.5, 0.1),
            [x[0] for x in errors],
            linewidth=2,
            label=model_name,
        )
        plt.fill_between(
            np.arange(0, 4.5, 0.1),
            [x[0] - x[1] for x in errors],
            [x[0] + x[1] for x in errors],
            alpha=0.2,
        )

    plt.xlabel("Min distance from sample rating to avg user rating")
    plt.ylabel("RMSE")

    plt.xlim([0, 4])
    plt.ylim([0, 4])

    plt.xticks(np.arange(0, 4.5, 0.5))
    plt.yticks(np.arange(0, 4.5, 0.5))

    plt.grid(which="major", axis="both", linestyle="-", linewidth=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()


# 2x2 grid of plots
# Upper plots are the train samples, lower plots are the test samples
# Left plots are the uncorrected predictions, right plots are the corrected predictions
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, samples in enumerate([train_samples_selection, test_samples_selection]):
    # Group by rating and calculate the mean and standard deviation of the predictions (not the pred_bin)
    df = samples.groupby("rating")["pred"].agg(["mean", "std"]).reset_index()

    # Make sure the dataframe has all the ratings
    df = df.merge(pd.DataFrame({"rating": samples["rating"].unique()}), on="rating", how="outer")

    # Plot the mean and standard deviation of the predictions (circle and area)
    axs[i, 0].plot(df["rating"], df["mean"], color="blue", linewidth=1, marker="o", markersize=3)
    axs[i, 0].fill_between(
        df["rating"],
        df["mean"] - df["std"],
        df["mean"] + df["std"],
        alpha=0.2,
        color="blue",
    )

    # Plot the x=y line
    axs[i, 0].plot(
        [1, samples["rating"].max()],
        [1, samples["pred"].max()],
        color="black",
        linewidth=1,
    )

    # Set the x and y limits
    axs[i, 0].set_xlim([1, samples["rating"].max()])
    axs[i, 0].set_ylim([1, samples["pred"].max()])

    # Set the x and y labels
    axs[i, 0].set_xlabel("Actual Rating")
    axs[i, 0].set_ylabel("Predicted Rating")

    df = samples.groupby("rating")["pred_corrected"].agg(["mean", "std"]).reset_index()

    # Make sure the dataframe has all the ratings
    df = df.merge(pd.DataFrame({"rating": samples["rating"].unique()}), on="rating", how="outer")

    # Make sure the values are floats
    df = df.astype("float")

    # Plot the mean and standard deviation of the predictions (circle and area)
    axs[i, 1].plot(df["rating"], df["mean"], color="blue", linewidth=1, marker="o", markersize=3)
    axs[i, 1].fill_between(
        df["rating"],
        df["mean"] - df["std"],
        df["mean"] + df["std"],
        alpha=0.2,
        color="blue",
    )

    # Plot the x=y line
    axs[i, 1].plot(
        [1, samples["rating"].max()],
        [1, samples["pred_corrected"].max()],
        color="black",
        linewidth=1,
    )

    # Set the x and y limits
    axs[i, 1].set_xlim([1, samples["rating"].max()])
    axs[i, 1].set_ylim([1, samples["pred_corrected"].max()])

    # Set the x and y labels
    axs[i, 1].set_xlabel("Actual Rating")
    axs[i, 1].set_ylabel("Predicted Rating")


plt.tight_layout()
plt.show()
