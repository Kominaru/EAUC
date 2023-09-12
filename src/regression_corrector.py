import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

MODEL_NAME = "MF"

train_samples: pd.DataFrame = pd.read_csv(f"outputs/{MODEL_NAME}/train_samples.csv")
test_samples: pd.DataFrame = pd.read_csv(
    f"outputs/{MODEL_NAME}/test_samples_with_predictions.csv"
)

# Print basic dataset statistics
all_samples = pd.concat([train_samples, test_samples])

print(f"Number of ratings: {len(all_samples)}")
print(f"Number of users: {len(all_samples['user_id'].unique())}")
print(f"Number of movies: {len(all_samples['movie_id'].unique())}")




def plot_error_distribution_by_difference_rating_to_avg_rating(samples, predicts_dict):
    samples["dist_avg_to_rating"] = (
        samples["user_avg_rating"] + samples["movie_avg_rating"]
    ) / 2 - samples["rating"]

    samples["dist_avg_to_rating_bin"] = pd.cut(
        samples["dist_avg_to_rating"], bins=np.arange(-5, 5, 0.25)
    )

    plt.figure(figsize=(7, 7))

    for model_name, predicts in predicts_dict.items():
        # error = samples["rating"] - predicts  # for Mean Error
        error = abs(samples["rating"] - predicts)  # for MAE
        # error = pow(samples["rating"] - predicts, 2)  # for RMSE

        samples["model_error"] = error

        # Get the RMSE for all samples where the distance is larger than min_dist
        filtered_samples = samples[samples["dist_avg_to_rating"] >= 1]
        rmse = np.sqrt(
            pow(filtered_samples['model_error'],2).mean()
        )
        print(f"RMSE for {model_name} in samples with dist > 1: {rmse:.3f}")

        # Compute the RMSE and std error for each bin
        bin_errors = (
            samples.groupby("dist_avg_to_rating_bin")["model_error"]
            .agg(["mean", "std"])
            .reset_index()
        )

        # SQRT if RMSE
        # bin_errors["mean"] = np.sqrt(bin_errors["mean"])

        # Use the middle of the bin as the x coordinate
        xx = bin_errors["dist_avg_to_rating_bin"].apply(lambda x: x.mid)

        # Plot the errors for the model
        plt.plot(xx, bin_errors["mean"], linewidth=2, label=model_name)
        plt.fill_between(
            xx,
            bin_errors["mean"] - bin_errors["std"],
            bin_errors["mean"] + bin_errors["std"],
            alpha=0.2,
        )

    # Plot the x=0 line (perfect regressor)
    plt.plot([-5, 5], [0, 0])

    # Add grid every 0.5 x or y increment, but only add axis ticks every 1 increment
    plt.grid(which="major", axis="both", linestyle="-", linewidth=0.5)
    plt.grid(which="minor", axis="both", linestyle=":", linewidth=0.5)

    # Configure minor ticks every 0.25
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    # Set x and y limits

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    # # Set x and y ticks every 0.5
    plt.xticks(np.arange(-5, 5, 1))
    plt.yticks(np.arange(-5, 5, 1))

    plt.xlabel("Avg(avgrating(user),avgrating(movie)) - rating")
    plt.ylabel("Prediction error")

    plt.legend()
    plt.tight_layout()
    plt.show()


# Compute the average rating per user and per movie
user_avg_ratings = train_samples.groupby("user_id")["rating"].mean()
movie_avg_ratings = train_samples.groupby("movie_id")["rating"].mean()

# Add the avg. ratings as a column to the train and test samples
train_samples["user_avg_rating"] = train_samples["user_id"].map(user_avg_ratings)
train_samples["movie_avg_rating"] = train_samples["movie_id"].map(movie_avg_ratings)

test_samples["user_avg_rating"] = test_samples["user_id"].map(user_avg_ratings)
test_samples["movie_avg_rating"] = test_samples["movie_id"].map(movie_avg_ratings)

# Make a selection of the train and test samples: those from users and movies
# with mean ratings in the range [4, 4.5]


def filter_by_avg_rating(samples, user_avg_rating_range, movie_avg_rating_range):
    # Warning, if a movie or user has no ratings in the train set, it will be filtered out
    # as an implicit filter by the user/movie average rating range
    return samples[
        (samples["user_avg_rating"] >= user_avg_rating_range[0])
        & (samples["user_avg_rating"] <= user_avg_rating_range[1])
        & (samples["movie_avg_rating"] >= movie_avg_rating_range[0])
        & (samples["movie_avg_rating"] <= movie_avg_rating_range[1])
    ]


train_samples_selection = filter_by_avg_rating(
    train_samples, user_avg_rating_range=[0, 5], movie_avg_rating_range=[0, 5]
)
test_samples_selection = filter_by_avg_rating(
    test_samples, user_avg_rating_range=[0, 5], movie_avg_rating_range=[0, 5]
)

print(f"Number of train samples: {len(train_samples_selection)}")
print(f"Number of test samples: {len(test_samples_selection)}")
print(f"Total number of samples: {len(train_samples_selection) + len(test_samples_selection)}")

# Make regression of the predictions on the ratings for the train samples, using linear and polynomial (degree 2) regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()

# Linear regression: based on the prediction, the user and movie average ratings, predict the real rating
lr.fit(
    train_samples_selection[["rating", "user_avg_rating", "movie_avg_rating"]],
    train_samples_selection["pred"],
)

# Print the equation of the linear regression
print(
    f"Linear regression: pred = {lr.intercept_:.3f} + {lr.coef_[0]:.3f} * rating + {lr.coef_[1]:.3f} * user_avg_rating + {lr.coef_[2]:.3f} * movie_avg_rating"
)


# If the regression is pred = a + b * rating + c * user_avg_rating + d * movie_avg_rating,
# then rating = (pred - a - c * user_avg_rating - d * movie_avg_rating) / b
def correct_prediction(sample, lr):
    return (
        sample["pred"]
        - lr.intercept_
        - lr.coef_[1] * sample["user_avg_rating"]
        - lr.coef_[2] * sample["movie_avg_rating"]
    ) / lr.coef_[0]


train_samples_selection["pred_corrected"] = train_samples_selection.apply(
    lambda x: correct_prediction(x, lr), axis=1
)
test_samples_selection["pred_corrected"] = test_samples_selection.apply(
    lambda x: correct_prediction(x, lr), axis=1
)

# Clip the corrected predictions to the range [1, 5]
train_samples_selection["pred_corrected"] = train_samples_selection[
    "pred_corrected"
].clip(1, 5)
test_samples_selection["pred_corrected"] = test_samples_selection[
    "pred_corrected"
].clip(1, 5)

print(
    "Train RMSE (uncorrected): ",
    mean_squared_error(
        train_samples_selection["rating"],
        train_samples_selection["pred"],
        squared=False,
    ),
)
print(
    "Test RMSE (uncorrected): ",
    mean_squared_error(
        test_samples_selection["rating"], test_samples_selection["pred"], squared=False
    ),
)
print(
    "Train RMSE (corrected): ",
    mean_squared_error(
        train_samples_selection["rating"],
        train_samples_selection["pred_corrected"],
        squared=False,
    ),
)
print(
    "Test RMSE (corrected): ",
    mean_squared_error(
        test_samples_selection["rating"],
        test_samples_selection["pred_corrected"],
        squared=False,
    ),
)

# Store the different predictions in a dictionary
predicts_dict = {
    "MF": train_samples_selection["pred"],
    "MF_corrected": train_samples_selection["pred_corrected"],
    "RND": np.random.uniform(1, 5, len(train_samples_selection)),
}


plot_error_distribution_by_difference_rating_to_avg_rating(
    train_samples_selection, predicts_dict
)

# Save the train and test samples with the corrected predictions. We overwrite the pred column with the corrected predictions
train_samples_selection["pred"] = train_samples_selection["pred_corrected"]
test_samples_selection["pred"] = test_samples_selection["pred_corrected"]

# Save only the columns we need
os.makedirs(f"outputs/{MODEL_NAME}_corrected", exist_ok=True)

train_samples_selection[["user_id", "movie_id", "rating", "pred"]].to_csv(
    f"outputs/{MODEL_NAME}_corrected/train_samples.csv", index=False
)

test_samples_selection[["user_id", "movie_id", "rating", "pred"]].to_csv(
    f"outputs/{MODEL_NAME}_corrected/test_samples_with_predictions.csv", index=False
)

exit()

# 2x2 grid of plots
# Upper plots are the train samples, lower plots are the test samples
# Left plots are the uncorrected predictions, right plots are the corrected predictions
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, samples in enumerate([train_samples_selection, test_samples_selection]):
    # Group by rating and calculate the mean and standard deviation of the predictions (not the pred_bin)
    df = samples.groupby("rating")["pred"].agg(["mean", "std"]).reset_index()

    # Make sure the dataframe has all the ratings
    df = df.merge(
        pd.DataFrame({"rating": samples["rating"].unique()}), on="rating", how="outer"
    )

    # Plot the mean and standard deviation of the predictions (circle and area)
    axs[i, 0].plot(
        df["rating"], df["mean"], color="blue", linewidth=1, marker="o", markersize=3
    )
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
    df = df.merge(
        pd.DataFrame({"rating": samples["rating"].unique()}), on="rating", how="outer"
    )

    # Make sure the values are floats
    df = df.astype("float")

    # Plot the mean and standard deviation of the predictions (circle and area)
    axs[i, 1].plot(
        df["rating"], df["mean"], color="blue", linewidth=1, marker="o", markersize=3
    )
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
