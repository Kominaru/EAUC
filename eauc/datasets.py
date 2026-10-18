"""Loaders of the datasets of the paper, all returning ``user_id, item_id, rating`` DataFrames.

The public datasets can be fetched with ``download_data.py``; GDSC1, CTRPv2, IMF DOTS and Kiva are read from
preprocessed files placed under ``<data_dir>/<dataset>/``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _as_codes(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Replace the given id columns by contiguous integer codes starting at 0."""
    for column in columns:
        df[column] = df[column].astype("category").cat.codes
    return df


def load_tripadvisor(data_dir: str, dataset_name: str) -> pd.DataFrame:
    """TripAdvisor reviews (https://zenodo.org/record/5644892), keeping users and items with >= 10 ratings."""
    df = pd.read_pickle(os.path.join(data_dir, dataset_name, "reviews.pkl"))
    df = df[["userId", "restaurantId", "rating"]]
    df.columns = ["user_id", "item_id", "rating"]

    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    df = df.groupby("item_id").filter(lambda x: len(x) >= 10)
    df = df.groupby("user_id").filter(lambda x: len(x) >= 10)
    df = df.dropna()

    df = _as_codes(df, "user_id", "item_id")
    df["rating"] = df["rating"] / 10
    return df


def load_movielens(data_dir: str, dataset_name: str) -> pd.DataFrame:
    """MovieLens 100K, 1M or 10M."""
    if dataset_name == "ml-100k":
        df = pd.read_csv(os.path.join(data_dir, dataset_name, "u.data"), sep="\t", header=None)
    elif dataset_name in ("ml-1m", "ml-10m"):
        df = pd.read_csv(os.path.join(data_dir, dataset_name, "ratings.dat"), sep="::", engine="python", header=None)
    else:
        raise ValueError(f"Unknown MovieLens dataset {dataset_name!r}")
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    return df[["user_id", "item_id", "rating"]]


def load_douban_monti(data_dir: str, dataset_name: str):
    """Douban (Monti et al.). Returns ``(all, train, test)`` following the dataset's own split."""
    import h5py

    with h5py.File(os.path.join(data_dir, dataset_name, "training_test_dataset.mat"), "r") as f:
        ratings = np.array(f["M"])
        train = ratings * np.array(f["Otraining"])
        test = ratings * np.array(f["Otest"])

    def to_frame(matrix: np.ndarray) -> pd.DataFrame:
        users, items = np.nonzero(matrix)
        return pd.DataFrame({"user_id": users, "item_id": items, "rating": matrix[users, items]})

    return to_frame(ratings), to_frame(train), to_frame(test)


def load_netflix_prize(data_dir: str) -> pd.DataFrame:
    """Netflix Prize ratings, from the ``ratings.csv`` created by ``download_data.py``."""
    df = pd.read_csv(os.path.join(data_dir, "netflix-prize", "ratings.csv"))
    df.columns = ["user_id", "item_id", "rating"]
    df = _as_codes(df, "user_id", "item_id")
    df["rating"] = df["rating"].astype(np.float32)
    return df


def load_gdsc1(data_dir: str) -> pd.DataFrame:
    """GDSC1 drug responses (ln IC50), subsampled to 30% of the data as in the paper."""
    df = pd.read_csv(os.path.join(data_dir, "gdsc1", "gdsc1_raw.csv"), sep=";")
    df = df[["COSMIC_ID", "DRUG_ID", "LN_IC50"]]
    df.columns = ["user_id", "item_id", "rating"]
    df = _as_codes(df.dropna(), "user_id", "item_id")
    return df.sample(frac=0.3, random_state=42).reset_index(drop=True)


def load_ctrpv2(data_dir: str) -> pd.DataFrame:
    """CTRPv2 drug responses, from a preprocessed ``ctrpv2_processed.csv``."""
    df = pd.read_csv(os.path.join(data_dir, "ctrpv2", "ctrpv2_processed.csv"))
    return _as_codes(df, "user_id", "item_id")


def load_dot_2023(data_dir: str) -> pd.DataFrame:
    """IMF DOTS 2023 trade values, from a preprocessed ``dot_2023_processed.csv``."""
    df = pd.read_csv(os.path.join(data_dir, "dot_2023", "dot_2023_processed.csv"))
    df["rating"] = df["rating"].astype(np.float32)
    return df


def load_kiva(data_dir: str) -> pd.DataFrame:
    """Kiva microloans, from a preprocessed ``user_profiles.csv``."""
    df = pd.read_csv(os.path.join(data_dir, "kiva-ml-17", "user_profiles.csv"))
    return _as_codes(df, "user_id")


def train_test_split(df: pd.DataFrame, test_size: float = 0.1, seed: int | None = None):
    """Random holdout split into ``(train, test)``; ``seed=None`` gives a fresh split on every call."""
    in_train = np.random.default_rng(seed).random(len(df)) < (1 - test_size)
    return df[in_train], df[~in_train]


def load_dataset(dataset_name: str, data_dir: str = "data") -> pd.DataFrame:
    """Load the full dataset ``dataset_name``.

    Supported names: ``ml-100k``, ``ml-1m``, ``ml-10m``, ``tripadvisor-*``, ``netflix-prize``,
    ``douban-monti``, ``gdsc1``, ``ctrpv2``, ``dot_2023`` and ``kiva-ml-17``.
    """
    if dataset_name.startswith("ml-"):
        return load_movielens(data_dir, dataset_name)
    if dataset_name.startswith("tripadvisor-"):
        return load_tripadvisor(data_dir, dataset_name)
    if dataset_name == "douban-monti":
        return load_douban_monti(data_dir, dataset_name)[0]
    loaders = {
        "netflix-prize": load_netflix_prize,
        "gdsc1": load_gdsc1,
        "ctrpv2": load_ctrpv2,
        "dot_2023": load_dot_2023,
        "kiva-ml-17": load_kiva,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset {dataset_name!r}")
    return loaders[dataset_name](data_dir)
