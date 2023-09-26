import matplotlib.pyplot as plt
import pandas as pd


def plot_dataset_statistics_figures(
    all_samples: pd.DataFrame, train_samples: pd.DataFrame, MODEL_NAME: pd.DataFrame
) -> None:
    """
    Plots the basic dataset statistics figures:
    - Histogram of the ratings as a whole with logarithmic y-scale
    - Histogram of the average rating per item (train set)
    - Histogram of the average rating per user (train set)
    :param all_samples: pandas dataframe containing all samples
    :param train_samples: pandas dataframe containing the train samples
    :param MODEL_NAME: name of the model
    :return: None
    """

    # 1. Plot histogram of the ratings as a whole with logarithmic y-scale (using bar plot)
    all_samples["rating"].value_counts().sort_index().plot.bar(logy=True, title="Histogram of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("# of ratings")
    plt.savefig("figures/" + MODEL_NAME + "/hist_ratings.pdf")
    plt.clf()

    # 2. Histogram of the average rating per item (train set)
    train_samples.groupby("item_id")["rating"].mean().plot.hist(
        bins=25, title="Histogram of Average Item Ratings", logy=True
    )
    plt.xlabel("Average Rating")
    plt.ylabel("# of items")
    plt.savefig("figures/" + MODEL_NAME + "/hist_avg_item_ratings.pdf")
    plt.clf()

    # 3. Histogram of the average rating per user (train set)
    train_samples.groupby("user_id")["rating"].mean().plot.hist(
        bins=25, title="Histogram of Average User Ratings", logy=True
    )
    plt.xlabel("Average Rating")
    plt.ylabel("# of users")
    plt.savefig("figures/" + MODEL_NAME + "/hist_avg_user_ratings.pdf")
    plt.clf()
