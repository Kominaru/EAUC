import argparse
import os
import pickle
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

import myfm
from myfm import RelationBlock
from myfm.gibbs import MyFMGibbsRegressor, MyFMOrderedProbit
from myfm.utils.benchmark_data.movielens100k_data import MovieLens100kDataManager
from myfm.utils.callbacks import (
    LibFMLikeCallbackBase,
    OrderedProbitCallback,
    RegressionCallback,
)
from myfm.utils.encoders import CategoryValueToSparseEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    This script apply the method and evaluation protocal proposed in
    "On the Difficulty of Evaluating Baselines" paper by Rendle et al,
    against smaller Movielens 100K dataset, using myFM.
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--fold_index",
        type=int,
        help="which index set to use as a test within 5-fold predefined CV.",
        default=1,
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        choices=["regression", "oprobit"],
        default="regression",
        help="specify the output type.",
    )
    parser.add_argument("-i", "--iteration", type=int, help="mcmc iteration", default=512)
    parser.add_argument("-d", "--dimension", type=int, help="fm embedding dimension", default=10)

    parser.add_argument(
        "--stricter_protocol",
        action="store_true",
        help="Whether to use the \"stricter\" protocol (i.e., don't include the test set implicit information) stated in [Rendle, '19].",
        default=True,
    )

    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        choices=["mf", "svdpp", "timesvd", "timesvdpp", "timesvdpp_flipped"],
        help="feature set used in the experiment.",
        default="timesvdpp_flipped",
    )

    args = parser.parse_args()

    random_seed = 42

    # Additional features.
    # We add
    # 1. date of evaluation as categorical variables
    # 2. "all users who have evaluated a movie in the train set" or
    # 3. "all movies rated by a user" as a feature of user/movie.
    if args.feature == "mf":
        use_date = False
        use_iu = False
        use_ii = False
    elif args.feature == "svdpp":
        use_date = False
        use_iu = True
        use_ii = False
    elif args.feature == "timesvd":
        use_date = True
        use_iu = False
        use_ii = False
    elif args.feature == "timesvdpp":
        use_date = True
        use_iu = True
        use_ii = False
    elif args.feature == "timesvdpp_flipped":
        use_date = True  # use date info or not
        use_iu = True  # use implicit user feature
        use_ii = True  # use implicit item feature
    else:
        raise ValueError("unknown feature set specified.")

    FOLD_INDEX = args.fold_index
    ITERATION = args.iteration
    DIMENSION = args.dimension
    if FOLD_INDEX < 1 or FOLD_INDEX >= 6:
        raise ValueError("fold_index must be in the range(1, 6).")
    ALGORITHM = args.algorithm

    # df = pd.read_csv("data/ml-100k/u.data", sep="\t", header=None)
    # df.columns = ["user_id", "movie_id", "rating", "timestamp"]

    def load_and_format_doubanmonti_data(dataset_name):
        # Open douban-monti dataset (matlab file) using h5py
        import h5py

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
        all_df = rating_matrix_to_dataframe(data)

        return all_df

    DATASET = "douban-monti"

    if DATASET == "gdsc1":
        df = pd.read_csv("data/gdsc1/gdsc1_processed.csv")
        df = df.sample(frac=0.3).reset_index(drop=True)
    elif DATASET == "dot_2023":
        df = pd.read_csv("data/dot_2023/dot_2023_processed.csv")
    elif DATASET == "CTRPv2":
        df = pd.read_csv("data/CTRPv2/CTRPv2_processed.csv")
    elif DATASET == "kiva-ml-17":
        df = pd.read_csv("data/kiva-ml-17/user_profiles.csv")
        df["user_id"] = df["user_id"].astype("category").cat.codes
    elif DATASET == "douban-monti":
        df = load_and_format_doubanmonti_data(DATASET)

    df = df[["user_id", "item_id", "rating"]]
    df.columns = ["user_id", "movie_id", "rating"]

    df_train = df.sample(frac=0.9)
    df_test = df.drop(df_train.index)
    # data_manager = MovieLens100kDataManager()
    # df_train, df_test = data_manager.load_rating_predefined_split(fold=FOLD_INDEX)

    if ALGORITHM == "oprobit":
        # interpret the rating (1, 2, 3, 4, 5) as class (0, 1, 2, 3, 4).
        for df_ in [df_train, df_test]:
            df_["rating"] -= 1
            df_["rating"] = df_.rating.astype(np.int32)

    if args.stricter_protocol:
        implicit_data_source = df_train
    else:
        implicit_data_source = pd.concat([df_train, df_test])

    user_to_internal = CategoryValueToSparseEncoder[int](implicit_data_source.user_id.values)
    movie_to_internal = CategoryValueToSparseEncoder[int](implicit_data_source.movie_id.values)

    print("df_train.shape = {}, df_test.shape = {}".format(df_train.shape, df_test.shape))
    # treat the days of events as categorical variable
    # date_encoder = CategoryValueToSparseEncoder(implicit_data_source.timestamp.dt.date.values)

    # def categorize_date(df: pd.DataFrame) -> sps.csr_matrix:
    #     return date_encoder.to_sparse(df.timestamp.dt.date.values)

    movie_vs_watched: Dict[int, List[int]] = dict()
    user_vs_watched: Dict[int, List[int]] = dict()

    for row in implicit_data_source.itertuples():
        user_id: int = row.user_id
        movie_id: int = row.movie_id
        movie_vs_watched.setdefault(movie_id, list()).append(user_id)
        user_vs_watched.setdefault(user_id, list()).append(movie_id)

    if use_date:
        X_date_train = categorize_date(df_train)
        X_date_test = categorize_date(df_test)
    else:
        X_date_train, X_date_test = (None, None)

    # setup grouping
    feature_group_sizes = []
    if use_date:
        feature_group_sizes.append(
            len(date_encoder),  # date
        )

    feature_group_sizes.append(len(user_to_internal))  # user ids

    if use_iu:
        # all movies which a user watched
        feature_group_sizes.append(len(movie_to_internal))

    feature_group_sizes.append(len(movie_to_internal))  # movie ids

    if use_ii:
        feature_group_sizes.append(len(user_to_internal))  # all the users who watched a movies

    grouping = [i for i, size in enumerate(feature_group_sizes) for _ in range(size)]

    # given user/movie ids, add additional infos and return it as sparse
    def augment_user_id(user_ids: List[int]) -> sps.csr_matrix:
        X = user_to_internal.to_sparse(user_ids)
        if not use_iu:
            return X
        data: List[float] = []
        row: List[int] = []
        col: List[int] = []
        for index, user_id in enumerate(user_ids):
            watched_movies = user_vs_watched.get(user_id, [])
            normalizer = 1 / max(len(watched_movies), 1) ** 0.5
            for mid in watched_movies:
                data.append(normalizer)
                col.append(movie_to_internal[mid])
                row.append(index)
        return sps.hstack(
            [
                X,
                sps.csr_matrix(
                    (data, (row, col)),
                    shape=(len(user_ids), len(movie_to_internal)),
                ),
            ],
            format="csr",
        )

    def augment_movie_id(movie_ids: List[int]) -> sps.csr_matrix:
        X = movie_to_internal.to_sparse(movie_ids)
        if not use_ii:
            return X

        data: List[float] = []
        row: List[int] = []
        col: List[int] = []

        for index, movie_id in enumerate(movie_ids):
            watched_users = movie_vs_watched.get(movie_id, [])
            normalizer = 1 / max(len(watched_users), 1) ** 0.5
            for uid in watched_users:
                data.append(normalizer)
                row.append(index)
                col.append(user_to_internal[uid])
        return sps.hstack(
            [
                X,
                sps.csr_matrix(
                    (data, (row, col)),
                    shape=(len(movie_ids), len(user_to_internal)),
                ),
            ]
        )

    # Create RelationBlock.
    train_blocks: List[RelationBlock] = []
    test_blocks: List[RelationBlock] = []
    for source, target in [(df_train, train_blocks), (df_test, test_blocks)]:
        unique_users, user_map = np.unique(source.user_id, return_inverse=True)
        target.append(RelationBlock(user_map, augment_user_id(unique_users)))
        unique_movies, movie_map = np.unique(source.movie_id, return_inverse=True)
        target.append(RelationBlock(movie_map, augment_movie_id(unique_movies)))

    trace_path = "rmse_{0}_fold_{1}.csv".format(ALGORITHM, FOLD_INDEX)

    callback: LibFMLikeCallbackBase
    fm: Union[MyFMGibbsRegressor, MyFMOrderedProbit]
    if ALGORITHM == "regression":
        fm = myfm.MyFMRegressor(rank=DIMENSION)
        callback = RegressionCallback(
            n_iter=ITERATION,
            X_test=X_date_test,
            y_test=df_test.rating.values,
            X_rel_test=test_blocks,
            clip_min=df_train.rating.min(),
            clip_max=df_train.rating.max(),
            trace_path=trace_path,
        )
    else:
        fm = myfm.MyFMOrderedProbit(rank=DIMENSION)
        callback = OrderedProbitCallback(
            n_iter=ITERATION,
            X_test=X_date_test,
            y_test=df_test.rating.values,
            n_class=5,
            X_rel_test=test_blocks,
            trace_path=trace_path,
        )

    fm.fit(
        X_date_train,
        df_train.rating.values,
        X_rel=train_blocks,
        grouping=grouping,
        n_iter=ITERATION,
        n_kept_samples=ITERATION - 5,
        callback=callback,
    )

    if ALGORITHM == "regression":
        # Predict test set and compute RMSE
        y_pred = fm.predict(X_date_test, test_blocks)
    else:
        y_pred = fm.predict_proba(X_date_test, test_blocks)
        y_pred = y_pred.dot(np.arange(5))
        y_pred = np.clip(y_pred, 1, 5)
    rmse = np.sqrt(np.mean((y_pred - df_test.rating.values) ** 2))
    print("RMSE = {}".format(rmse))

    # Add pred column to df_test and df_train
    df_test["pred"] = y_pred
    df_train["pred"] = fm.predict(X_date_train, train_blocks)

    # Change the movie_id column to item_id
    df_test.rename(columns={"movie_id": "item_id"}, inplace=True)
    df_train.rename(columns={"movie_id": "item_id"}, inplace=True)

    # Save the result
    from save_model_outputs import save_model_outputs

    save_model_outputs(
        train_df=df_train,
        test_df=df_test,
        model_name="BAYESIAN_SVD++",
        dataset_name=DATASET,
        model_params={
            "iteration": ITERATION,
            "dimension": DIMENSION,
            "algorithm": ALGORITHM,
            "fold_index": FOLD_INDEX,
            "feature": args.feature,
        },
    )

    with open("callback_result_{0}_fold_{1}.pkl".format(ALGORITHM, FOLD_INDEX), "wb") as ofs:
        pickle.dump(callback, ofs)
