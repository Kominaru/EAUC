import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


def load_and_format_tripadvisor_data(dataset_name):
    """
    Formats a raw TripAdvisor dataset into a the standard user/item format used in this project.
    The datasets are available at https://zenodo.org/record/5644892#.YmZ3Z2gzZPY

    Args:
        dataset_name (str): Name of the dataset (e.g. tripadvisor-paris)

    Returns:
        pandas.DataFrame: DataFrame containing the formatted dataset with the columns ['user_id', 'item_id', 'rating']
    """

    df = pd.read_pickle(os.path.join("data", dataset_name, "reviews.pkl"))
    df = df[["userId", "restaurantId", "rating"]]
    df.columns = ["user_id", "item_id", "rating"]

    # Drop items with less than 100 ratings and users with less than 20 ratings
    # Remove repeated user-item pairs
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    df = df.groupby("item_id").filter(lambda x: len(x) >= 20)
    df = df.groupby("user_id").filter(lambda x: len(x) >= 5)

    # Remove NA values
    df = df.dropna()

    # Create new user and item ids ( userId's are strings, restaurantId's are not continuous)
    df["user_id"] = df["user_id"].astype("category").cat.codes
    df["item_id"] = df["item_id"].astype("category").cat.codes

    df["rating"] = df["rating"] / 10

    return df


def load_and_format_movielens_data(dataset_name):
    """
    Formats a raw MovieLens dataset into a the standard user/item format used in this project.

    Args:
        dataset_name (str): Name of the dataset (e.g. ml-1m)

    Returns:
        pandas.DataFrame: DataFrame containing the formatted dataset with the columns ['user_id', 'item_id', 'rating']
    """

    df = pd.read_csv(os.path.join("data", dataset_name, "ratings.dat"), sep="::", engine="python", header=None)
    df.columns = ["user_id", "movie_id", "rating", "timestamp"]
    df = df[["user_id", "movie_id", "rating"]]

    return df


class DyadicRegressionDataset(Dataset):
    """
    Represents a dataset for regression over dyadic data.
    """

    def __init__(self, df):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing the dataset
                Must contain at least the columns ['user_id', 'item_id', 'rating']
        """

        self.data = df
        self.data["user_id"] = self.data["user_id"].astype(np.int64)
        self.data["item_id"] = self.data["item_id"].astype(np.int64)
        self.data["rating"] = self.data["rating"].astype(np.float32)

    def __len__(self):
        """
        Returns:
            int: Length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: Tuple containing the user_id, item_id and rating
        """

        user_id = self.data.at[idx, "user_id"]
        item_id = self.data.at[idx, "item_id"]
        rating = self.data.at[idx, "rating"]

        return user_id, item_id, rating


class DyadicRegressionDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=0, test_size=0.1, dataset_name="ml-1m"):
        """
        Creates a dyadic regression datamodule with a holdout train-test split.
        Downloads the dataset if it doesn't exist in the data directory.

        Args:
            data_dir (str): Directory where the dataset is stored
            batch_size (int): Batch size
            num_workers (int): Number of workers for the DataLoader
            test_size (float): Fraction of the dataset to be used as test set
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size

        # Load the dataset from the file
        if dataset_name.startswith("ml-"):
            self.data = load_and_format_movielens_data(dataset_name)
        elif dataset_name.startswith("tripadvisor-"):
            self.data = load_and_format_tripadvisor_data(dataset_name)

        # Split the df into train and test sets (pandas dataframe)

        msk = np.random.rand(len(self.data)) < (1 - self.test_size)

        self.train_df = self.data[msk]
        self.test_df = self.data[~msk]

        print(f"#Training samples: {len(self.train_df)}")
        print(f"#Test samples    : {len(self.test_df)}")

        train_tuples = self.train_df[["user_id", "item_id"]].apply(tuple, axis=1)
        test_tuples = self.test_df[["user_id", "item_id"]].apply(tuple, axis=1)

        print("#Repeated samples: ", test_tuples.isin(train_tuples).sum())

        # Calculate the number of users and items in the dataset
        self.num_users = self.data["user_id"].max() + 1
        self.num_items = self.data["item_id"].max() + 1
        self.mean_rating = self.data["rating"].mean()

        # Reset index and create PyTorch datasets

        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        self.train_dataset = DyadicRegressionDataset(self.train_df)
        self.test_dataset = DyadicRegressionDataset(self.test_df)

    def train_dataloader(self):
        """
        Returns:
            torch.utils.data.DataLoader: DataLoader for the training set
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Returns:
            torch.utils.data.DataLoader: DataLoader for the validation set (same as test set)
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )

    def test_dataloader(self):
        """
        Returns:
            torch.utils.data.DataLoader: DataLoader for the test set
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )
