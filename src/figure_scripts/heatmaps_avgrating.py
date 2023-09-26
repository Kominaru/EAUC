import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_ratings_vs_preds_2dheatmaps_grid(
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    model_name: str,
    preds_bin_interval: int = 0.5,
    avgs_bin_interval: int = 0.5,
) -> None:
    # If train and test are the same, then we add a flag for later
    train_test_same = train_samples.equals(test_samples)

    preds_bins = np.arange(test_samples["rating"].min(), 5 + preds_bin_interval, preds_bin_interval)
    avg_ratings_bins = np.arange(test_samples["rating"].min(), 5 + avgs_bin_interval, avgs_bin_interval)

    user_avg_ratings = train_samples.groupby("user_id")["rating"].mean()
    item_avg_ratings = train_samples.groupby("item_id")["rating"].mean()

    test_samples["user_avg_rating"] = test_samples["user_id"].map(user_avg_ratings)
    test_samples["item_avg_rating"] = test_samples["item_id"].map(item_avg_ratings)

    test_samples["user_bin"] = pd.cut(test_samples["user_avg_rating"], bins=avg_ratings_bins, include_lowest=True)
    test_samples["item_bin"] = pd.cut(test_samples["item_avg_rating"], bins=avg_ratings_bins, include_lowest=True)

    user_bins = list(test_samples["user_bin"].unique().categories)

    # Precompute the logarithmic frequency for all combinations of rating and prediction
    test_samples["pred_bin"] = pd.cut(test_samples["pred"], bins=preds_bins, include_lowest=True)

    frequency_table = (
        test_samples.groupby(["user_bin", "item_bin", "rating", "pred_bin"]).size().reset_index(name="frequency")
    )
    frequency_table["log_frequency"] = frequency_table["frequency"].apply(lambda x: np.NaN if x == 0 else np.log10(x))

    # Create a grid of plots
    fig, axs = plt.subplots(len(user_bins), len(user_bins), figsize=(5 * (len(user_bins)), 5 * (len(user_bins))))

    # For each combination of user_bin and item_bin, create a heatmap of the frequency of ratings vs the prediction
    for i, user_bin in enumerate(user_bins):
        for j, item_bin in enumerate(user_bins):
            df = frequency_table[(frequency_table["user_bin"] == user_bin) & (frequency_table["item_bin"] == item_bin)]

            # If all frequencies are NaN, skip this subplot making it fully white
            if df["log_frequency"].isna().all():
                axs[i, j].axis("off")
                continue

            # Create the pivot table to reshape the data for the heatmap
            pivot_table = df.pivot(index="pred_bin", columns="rating", values="log_frequency")

            # Create the heatmap
            sns.heatmap(
                pivot_table,
                cmap="Blues",
                cbar_kws={"label": "log10(frequency)"},
                ax=axs[i, j],
                vmin=0,
                vmax=frequency_table["log_frequency"].max(),
            )
            axs[i, j].set_xlabel("Actual Rating")
            axs[i, j].set_ylabel("Predicted Rating")
            axs[i, j].set_title(f"user avg {user_bin}\n item avg {item_bin}")

            # Make sure the heatmap is square by making each subplot have square axes
            axs[i, j].set_box_aspect(1)

            # Flip on the y-axis to have (0,0) in the bottom left corner
            axs[i, j].invert_yaxis()

            # Remove colorbar
            axs[i, j].collections[0].colorbar.remove()

            # Plot an x=y line
            axs[i, j].plot(
                [0, test_samples["rating"].nunique()],
                [0, test_samples["pred_bin"].nunique()],
                color="black",
                linewidth=1,
            )

    plt.tight_layout()
    plt.savefig("figures/" + model_name + "/ratings_vs_preds_2d_heatmaps_grid.pdf")

    # Second figure: for each user and item, plot a line plot of the rating vs average prediction and standard deviation of predictions
    fig, axs = plt.subplots(len(user_bins), len(user_bins), figsize=(5 * (len(user_bins)), 5 * (len(user_bins))))

    for i, user_bin in enumerate(user_bins):
        for j, item_bin in enumerate(user_bins):
            # Select the samples inside the current user_bin and item_bin
            df = test_samples[(test_samples["user_bin"] == user_bin) & (test_samples["item_bin"] == item_bin)]

            # If there are no samples, skip this subplot
            if len(df) == 0:
                axs[i, j].axis("off")
                continue

            # Group by rating and calculate the mean and standard deviation of the predictions (not the pred_bin)

            df = df.groupby("rating")["pred"].agg(["mean", "std"]).reset_index()

            # Make sure the dataframe has all the ratings
            df = df.merge(pd.DataFrame({"rating": test_samples["rating"].unique()}), on="rating", how="outer")

            # Plot the mean and standard deviation of the predictions (circle and area)
            axs[i, j].plot(df["rating"], df["mean"], color="blue", linewidth=1, marker="o", markersize=3)
            axs[i, j].fill_between(
                df["rating"], df["mean"] - df["std"], df["mean"] + df["std"], alpha=0.2, color="blue"
            )

            # Plot the x=y line
            axs[i, j].plot(
                [1, test_samples["rating"].max()], [1, test_samples["pred"].max()], color="black", linewidth=1
            )

            # Set the x and y limits
            axs[i, j].set_xlim([1, test_samples["rating"].max()])
            axs[i, j].set_ylim([1, test_samples["pred"].max()])

            # Set the x and y labels
            axs[i, j].set_xlabel("Actual Rating")
            axs[i, j].set_ylabel("Predicted Rating")

            # Plot a red dot at the average of the average user and item ratings
            axs[i, j].plot(
                [user_bin.mid * 0.5 + item_bin.mid * 0.5],
                [user_bin.mid * 0.5 + item_bin.mid * 0.5],
                color="red",
                marker="o",
                markersize=5,
            )

            # Set the title
            axs[i, j].set_title(f"user avg {user_bin}\n item avg {item_bin}")

    plt.tight_layout()

    if train_test_same:
        plt.savefig("figures/" + model_name + "/ratings_vs_preds_lineplots_grid_train.pdf")
    else:
        plt.savefig("figures/" + model_name + "/ratings_vs_preds_lineplots_grid.pdf")
