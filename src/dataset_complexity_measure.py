import pandas as pd
import numpy as np
from scipy.stats import uniform, kstest
import tqdm
import h5py
import os

# Load dataset
dataset_name = "tripadvisor-london"

if dataset_name == "ml-1m":
    df = pd.read_csv("data/ml-1m/ratings.dat", sep="::", engine="python")
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
elif dataset_name == "ml-100k":
    df = pd.read_csv("data/ml-100k/u.data", sep="\t", header=None)
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
elif dataset_name == "ml-10m":
    df = pd.read_csv("data/ml-10m/ratings.dat", sep="::", engine="python")
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
elif dataset_name == "douban-monti":
    with h5py.File(os.path.join("data", dataset_name, "training_test_dataset.mat"), "r") as f:
        # Convert to numpy arrays

        data = np.array(f["M"])
        train_data = np.array(f["M"]) * np.array(f["Otraining"])
        test_data = np.array(f["M"]) * np.array(f["Otest"])

    def rating_matrix_to_dataframe(ratings: np.ndarray):
        """
        Converts a rating matrix to a pandas DataFrame.

        Args:
            ratings (np.ndarray): Rating matrix

        Returns:
            pandas.DataFrame: DataFrame containing the ratings
        """

        # Get the indices of the non-zero ratings
        nonzero_indices = np.nonzero(ratings)

        # Create the dataframe

        df = pd.DataFrame(
            {
                "user_id": nonzero_indices[0],
                "item_id": nonzero_indices[1],
                "rating": ratings[nonzero_indices],
            }
        )

        # Min and max ratings
        min_rating = df["rating"].min()
        max_rating = df["rating"].max()

        return df

    # Convert the training and test data to dataframes
    df = rating_matrix_to_dataframe(data)
    train_df = rating_matrix_to_dataframe(train_data)
    test_df = rating_matrix_to_dataframe(test_data)

    df.columns = ["user_id", "item_id", "rating"]

elif dataset_name == "tripadvisor-london":
    df = pd.read_pickle(os.path.join("data", dataset_name, "reviews.pkl"))
    df = df[["userId", "restaurantId", "rating"]]
    df.columns = ["user_id", "item_id", "rating"]

    # Drop items with less than 100 ratings and users with less than 20 ratings
    # Remove repeated user-item pairs
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    df = df.groupby("item_id").filter(lambda x: len(x) >= 10)
    df = df.groupby("user_id").filter(lambda x: len(x) >= 10)

    # Remove NA values
    df = df.dropna()

    # Create new user and item ids ( userId's are strings, restaurantId's are not continuous)
    df["user_id"] = df["user_id"].astype("category").cat.codes
    df["item_id"] = df["item_id"].astype("category").cat.codes

    df["rating"] = df["rating"] / 10

elif dataset_name == "netflix-prize":
    # Load the data
    df = pd.read_csv("data/netflix-prize/ratings.csv")

    # Rename the columns
    df.columns = ["user_id", "item_id", "rating"]

    # Make the ids start from 0 by creating new user and item ids
    df["user_id"] = df["user_id"].astype("category").cat.codes
    df["item_id"] = df["item_id"].astype("category").cat.codes

    # Convert ratings to float
    df["rating"] = df["rating"].astype(np.float32)

# Rename columns to user_id, item_id, rating, and remove timestamp

a = df["rating"].min()
b = df["rating"].max()

print(uniform(loc=a, scale=b - a).cdf)


def measure_similarity_to_uniform(data_grouped):
    ks_stat, p_value = kstest(data_grouped, uniform(loc=a, scale=b - a).cdf)
    return (ks_stat, p_value)


tqdm.tqdm.pandas()


# Measure for each user
user_similarity_results = df.groupby("user_id")["rating"].progress_apply(lambda x: measure_similarity_to_uniform(x))

# Measure for each item
item_similarity_results = df.groupby("item_id")["rating"].progress_apply(lambda x: measure_similarity_to_uniform(x))

# Concat results and obtain average and std of both ks_stat and p_value
results = pd.concat([user_similarity_results, item_similarity_results])
avg_ks_stat = results.apply(lambda x: x[0]).mean()
avg_p_value = results.apply(lambda x: x[1]).mean()

std_ks_stat = results.apply(lambda x: x[0]).std()
std_p_value = results.apply(lambda x: x[1]).std()

print(f"Average +- std K-S Statistic: {avg_ks_stat:.6f} +- {std_ks_stat:.6f}")
print(f"Average +- std p-value: {avg_p_value:.6f} +- {std_p_value:.6f}")
