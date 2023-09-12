import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision.datasets.utils import download_url

class MovieLensDataset(Dataset):
    '''
    MovieLens dataset class
    '''
    def __init__(self, df):
        '''
        Args:
            df (pandas.DataFrame): DataFrame containing the dataset
                Must contain at least the columns ['user_id', 'movie_id', 'rating']
        '''
        

        self.data = df
        self.data['user_id'] = self.data['user_id'].astype(np.int64)
        self.data['movie_id'] = self.data['movie_id'].astype(np.int64)
        self.data['rating'] = self.data['rating'].astype(np.float32)

    def __len__(self):
        '''
        Returns:
            int: Length of the dataset
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Args:
            idx (int): Index

        Returns:
            tuple: Tuple containing the user_id, movie_id and rating
        '''

        user_id = self.data.at[idx, 'user_id']
        movie_id = self.data.at[idx, 'movie_id']
        rating = self.data.at[idx, 'rating']

        return user_id, movie_id, rating

class MovieLensDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=0, test_size=0.1):
        '''
        Creates a MovieLens datamodule with a holdout train-test split.
        Downloads the dataset if it doesn't exist in the data directory.

        Args:
            data_dir (str): Directory where the dataset is stored
            batch_size (int): Batch size
            num_workers (int): Number of workers for the DataLoader
            test_size (float): Fraction of the dataset to be used as test set
        '''
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size

        # Download the dataset if it doesn't exist in the data directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

            data_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
            download_url(data_url, self.data_dir, 'ml-1m.zip', None)
            import zipfile
            with zipfile.ZipFile(os.path.join(self.data_dir, 'ml-1m.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

        # Load the dataset from the file
        self.data = pd.read_csv(os.path.join(self.data_dir, "ml-1m/ratings.dat"), sep='::', engine='python', header=None)
        self.data.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

        # Split the df into train and test sets (pandas dataframe)

        msk = np.random.rand(len(self.data)) < (1 - self.test_size)

        self.train_df = self.data[msk]
        self.test_df = self.data[~msk]

        print(f"#Training samples: {len(self.train_df)}")
        print(f"#Test samples    : {len(self.test_df)}")

        train_tuples = self.train_df[['user_id', 'movie_id']].apply(tuple, axis=1)
        test_tuples = self.test_df[['user_id', 'movie_id']].apply(tuple, axis=1)

        print('#Repeated samples: ', test_tuples.isin(train_tuples).sum())

        # Calculate the number of users and movies in the dataset
        self.num_users = self.data['user_id'].max() + 1
        self.num_movies = self.data['movie_id'].max() + 1
        self.mean_rating = self.data['rating'].mean()

        # Reset index and create PyTorch datasets

        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        self.train_dataset = MovieLensDataset(self.train_df)
        self.test_dataset = MovieLensDataset(self.test_df)

    def train_dataloader(self):
        '''
        Returns:
            torch.utils.data.DataLoader: DataLoader for the training set
        '''
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        '''
        Returns:
            torch.utils.data.DataLoader: DataLoader for the validation set (same as test set)
        '''
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        '''
        Returns:
            torch.utils.data.DataLoader: DataLoader for the test set
        '''
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
