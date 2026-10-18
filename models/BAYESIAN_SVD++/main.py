"""Bayesian SVD++ baseline: a factorization machine with implicit user and item features, fitted with myFM.

Follows the evaluation protocol of "On the Difficulty of Evaluating Baselines" (Rendle et al.) using myFM's
Gibbs-sampling regressor: every user is also described by the items they rated and every item by the users
who rated it, using the train set only. Predictions are written to
``outputs/<dataset>/BAYESIAN_SVD++/{train,test}_outputs_<n>.csv``.

Run from the repository root, e.g. ``python models/BAYESIAN_SVD++/main.py --dataset ml-1m``.
"""

import argparse

import myfm
import numpy as np
from myfm import RelationBlock
from myfm.utils.encoders import CategoryValueToSparseEncoder
from scipy import sparse as sps

from eauc import save_model_outputs
from eauc.datasets import load_dataset, train_test_split

# Embedding dimension of the paper (Table V): 32 for MovieLens 1M, 10 for the other datasets
DEFAULT_DIMENSION = {"ml-1m": 32}
FALLBACK_DIMENSION = 10


def implicit_features(ids, encoder, other_encoder, watched):
    """Sparse block ``[one-hot id | normalised multi-hot of the other side's ids that interacted with it]``."""
    rows, cols, data = [], [], []
    for index, entity in enumerate(ids):
        neighbours = watched.get(entity, [])
        weight = 1 / max(len(neighbours), 1) ** 0.5
        for neighbour in neighbours:
            rows.append(index)
            cols.append(other_encoder[neighbour])
            data.append(weight)
    implicit = sps.csr_matrix((data, (rows, cols)), shape=(len(ids), len(other_encoder)))
    return sps.hstack([encoder.to_sparse(ids), implicit], format="csr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("-i", "--iteration", type=int, default=512, help="MCMC iterations.")
    parser.add_argument("-d", "--dimension", type=int, help="FM embedding dimension (default: the paper's).")
    parser.add_argument("--seed", type=int, help="Seed of the random 90/10 train-test split.")
    args = parser.parse_args()

    dimension = args.dimension or DEFAULT_DIMENSION.get(args.dataset, FALLBACK_DIMENSION)
    df_train, df_test = train_test_split(load_dataset(args.dataset, args.data_dir), test_size=0.1, seed=args.seed)
    df_train, df_test = df_train.copy(), df_test.copy()
    print(f"df_train.shape = {df_train.shape}, df_test.shape = {df_test.shape}")

    # Implicit features come from the train set only (the "stricter" protocol of Rendle et al.)
    user_encoder = CategoryValueToSparseEncoder[int](df_train.user_id.values)
    item_encoder = CategoryValueToSparseEncoder[int](df_train.item_id.values)
    items_of_user: dict[int, list[int]] = {}
    users_of_item: dict[int, list[int]] = {}
    for row in df_train.itertuples():
        items_of_user.setdefault(row.user_id, []).append(row.item_id)
        users_of_item.setdefault(row.item_id, []).append(row.user_id)

    def relation_blocks(df):
        users, user_index = np.unique(df.user_id, return_inverse=True)
        items, item_index = np.unique(df.item_id, return_inverse=True)
        return [
            RelationBlock(user_index, implicit_features(users, user_encoder, item_encoder, items_of_user)),
            RelationBlock(item_index, implicit_features(items, item_encoder, user_encoder, users_of_item)),
        ]

    train_blocks, test_blocks = relation_blocks(df_train), relation_blocks(df_test)

    # One feature group per block: user id, items of the user, item id, users of the item
    group_sizes = [len(user_encoder), len(item_encoder), len(item_encoder), len(user_encoder)]
    grouping = [group for group, size in enumerate(group_sizes) for _ in range(size)]

    fm = myfm.MyFMRegressor(rank=dimension)
    fm.fit(
        None,
        df_train.rating.values,
        X_rel=train_blocks,
        grouping=grouping,
        n_iter=args.iteration,
        n_kept_samples=args.iteration - 5,
    )

    # Predictions are clipped to the observed value range, as in myFM's regression callback
    low, high = df_train.rating.min(), df_train.rating.max()
    df_train["pred"] = np.clip(fm.predict(None, train_blocks, n_workers=8), low, high)
    df_test["pred"] = np.clip(fm.predict(None, test_blocks, n_workers=8), low, high)
    print(f"RMSE = {np.sqrt(np.mean((df_test.pred - df_test.rating) ** 2)):.4f}")

    save_model_outputs(
        df_train,
        df_test,
        "BAYESIAN_SVD++",
        args.dataset,
        {"iteration": args.iteration, "dimension": dimension},
    )


if __name__ == "__main__":
    main()
