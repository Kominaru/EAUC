"""Dataset difficulty for eccentricity bias: entity-wise Kolmogorov-Smirnov statistic D_KS (paper Eq. 6).

Loads a dataset with the loaders of the MF model script and reports the mean KS distance to a uniform
distribution over users, over items and over both. For your own data, ``eauc ks --data ratings.csv`` does
the same from a CSV file.

Run from the repository root, e.g.:
    python -m experiments.dataset_difficulty --dataset ml-1m
"""

import argparse

import pandas as pd

from eauc import entity_ks
from eauc.datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--clip", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Clip the observed values first.")
    args = parser.parse_args()

    df = load_dataset(args.dataset, args.data_dir)
    if args.clip:
        df["rating"] = df["rating"].clip(*args.clip)

    users, items = entity_ks(df, "user_id"), entity_ks(df, "item_id")
    for name, stats in (("users", users), ("items", items), ("all", pd.concat([users, items]))):
        print(f"K-S ({name}): {stats.mean():.4f} +- {stats.std():.4f}")


if __name__ == "__main__":
    main()
