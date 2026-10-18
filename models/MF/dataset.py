"""Torch dataset and Lightning data module for dyadic regression with the MF model."""

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from eauc.datasets import load_dataset, load_douban_monti, train_test_split


class DyadicRegressionDataset(Dataset):
    """Dataset of ``(user_id, item_id, rating)`` triplets backed by a DataFrame with a default index."""

    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.data["user_id"] = self.data["user_id"].astype(np.int64)
        self.data["item_id"] = self.data["item_id"].astype(np.int64)
        self.data["rating"] = self.data["rating"].astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data.at[idx, "user_id"],
            self.data.at[idx, "item_id"],
            self.data.at[idx, "rating"],
        )


class DyadicRegressionDataModule(LightningDataModule):
    """Loads a dataset and splits it into train and test sets (the test set also serves as validation set).

    Args:
        data_dir: Directory holding the datasets (see ``download_data.py``).
        dataset_name: One of ``ml-100k``, ``ml-1m``, ``ml-10m``, ``tripadvisor-*``, ``netflix-prize``,
            ``douban-monti``, ``gdsc1``, ``ctrpv2``, ``dot_2023``, ``kiva-ml-17``.
        batch_size: Batch size of the data loaders.
        num_workers: Number of data loader workers.
        test_size: Fraction of the data held out. Ignored for ``douban-monti``, which has its own split.
        seed: Seed of the random train-test split (``None`` for a fresh split on every run).
        verbose: Print dataset statistics.
    """

    def __init__(
        self, data_dir, dataset_name="ml-1m", batch_size=64, num_workers=0, test_size=0.1, seed=None, verbose=True
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        if dataset_name == "douban-monti":
            self.data, train_df, test_df = load_douban_monti(data_dir, dataset_name)
        else:
            self.data = load_dataset(dataset_name, data_dir)
            train_df, test_df = train_test_split(self.data, test_size, seed)

        self.num_users = self.data["user_id"].max() + 1
        self.num_items = self.data["item_id"].max() + 1
        self.min_rating = self.data["rating"].min()
        self.max_rating = self.data["rating"].max()

        if verbose:
            n_users, n_items = self.data["user_id"].nunique(), self.data["item_id"].nunique()
            print(f"#Training samples: {len(train_df)} | #Test samples: {len(test_df)}")
            print(f"#Users: {n_users} | #Items: {n_items} | Density: {len(self.data) / (n_users * n_items):.3f}")
            print(
                f"Rating range: {self.min_rating:.3f} - {self.max_rating:.3f} (mean {self.data['rating'].mean():.3f})"
            )

        self.train_dataset = DyadicRegressionDataset(train_df.reset_index(drop=True))
        self.test_dataset = DyadicRegressionDataset(test_df.reset_index(drop=True))

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
