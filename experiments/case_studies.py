"""Eccentric examples where an MF model is badly wrong (paper Fig. 6): MovieLens, CTRPv2 and IMF DOTS.

Reads run ``--run`` of the MF predictions in ``outputs/<dataset>/MF/`` and prints the examples whose observed
value is far from what the user and the item usually get, together with their metadata.

Run from the repository root:  python -m experiments.case_studies --run 1
"""

import argparse

import pandas as pd

from eauc import add_eccentricity, load_model_outputs


def movielens(train: pd.DataFrame, test: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Users and movies that are usually rated very high, with a rating of 1 or lower."""
    scored = add_eccentricity(train, test)
    cases = scored[(scored["user_mean"] > 4.25) & (scored["item_mean"] > 4.25) & (scored["rating"] <= 1)]
    movies = pd.read_csv(f"{data_dir}/ml-1m/movies.dat", sep="::", engine="python", header=None, encoding="latin-1")
    users = pd.read_csv(f"{data_dir}/ml-1m/users.dat", sep="::", engine="python", header=None, encoding="latin-1")
    return cases.merge(movies, left_on="item_id", right_on=0).merge(users, left_on="user_id", right_on=0)


def ctrpv2(train: pd.DataFrame, test: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Cell lines and drugs that usually respond weakly, with a very high response."""
    scored = add_eccentricity(train, test)
    cases = scored[(scored["user_mean"] < 13) & (scored["item_mean"] < 13) & (scored["rating"] > 18.5)]
    drugs = pd.read_csv(f"{data_dir}/ctrpv2/drug_id_mapping.csv")
    cell_lines = pd.read_csv(f"{data_dir}/ctrpv2/user_id_mapping.csv")
    return cases.merge(drugs, left_on="item_id", right_on="drug_id").merge(
        cell_lines, left_on="user_id", right_on="new_user_id"
    )


def dot_2023(train: pd.DataFrame, test: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Country pairs with very different average trade values and a large prediction error.

    Countries act as both users and items, so their average is taken over both roles.
    """
    totals = pd.concat(
        [
            train.groupby("user_id")["rating"].agg(["count", "sum"]),
            train.groupby("item_id")["rating"].agg(["count", "sum"]),
        ]
    )
    pooled = totals.groupby(level=0).sum()
    average = pooled["sum"] / pooled["count"]

    scored = test.assign(user_mean=test["user_id"].map(average), item_mean=test["item_id"].map(average))
    scored["avg_diff"] = (scored["user_mean"] - scored["item_mean"]).abs()
    scored["abs_error"] = (scored["rating"] - scored["pred"]).abs()
    cases = scored[(scored["avg_diff"] > 3) & (scored["abs_error"] > 1.5)]

    countries = pd.read_csv(f"{data_dir}/dot_2023/country_id_mapping.csv")
    return cases.merge(countries, left_on="user_id", right_on="new_id").merge(
        countries, left_on="item_id", right_on="new_id"
    )


STUDIES = {"ml-1m": movielens, "ctrpv2": ctrpv2, "dot_2023": dot_2023}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, default=1, help="Run number of the MF predictions.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--datasets", nargs="+", choices=list(STUDIES), default=list(STUDIES))
    args = parser.parse_args()

    for dataset in args.datasets:
        train, test = load_model_outputs("MF", dataset, args.run)
        print(f"\n{'#' * 10} {dataset}")
        print(STUDIES[dataset](train, test, args.data_dir).to_string())


if __name__ == "__main__":
    main()
