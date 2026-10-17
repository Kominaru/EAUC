"""Eccentricity-Area Under the Curve (EAUC) and related dyadic-regression diagnostics.

A *dyadic regression* dataset is a table of ``(user, item, value)`` observations (user-product ratings,
drug-cell line responses, country-to-country trade, ...). The functions here only need the observed
value and the model prediction of each row, so they apply to any such table.

Definitions (paper, Section IV):

* ``mean_u`` / ``mean_i``: mean observed value of each user / item in the *train* set.
* Dyadic Mean Value: ``DMV_ui = (mean_u + mean_i) / 2``.
* Eccentricity of an observation: ``Ecc_ui = |r_ui - DMV_ui|``.
* EAUC: area under the curve (eccentricity, absolute error) over the test set, normalised by
  ``(max(r) - min(r)) ** 2``. It lies in ``[0, 1]`` and lower is better.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

Method = Literal["ordered", "grouped", "bins"]


def _check_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required column(s): {missing}")


def add_eccentricity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    value_col: str = "rating",
) -> pd.DataFrame:
    """Return a copy of ``test`` with ``user_mean``, ``item_mean``, ``dmv`` and ``eccentricity`` columns.

    Means are computed on ``train`` only. Rows whose user or item does not appear in ``train`` are dropped,
    since their eccentricity is undefined.
    """
    _check_columns(train, [user_col, item_col, value_col], "train")
    _check_columns(test, [user_col, item_col, value_col], "test")

    user_means = train.groupby(user_col)[value_col].mean()
    item_means = train.groupby(item_col)[value_col].mean()

    out = test.copy()
    out["user_mean"] = out[user_col].map(user_means)
    out["item_mean"] = out[item_col].map(item_means)
    out = out[out["user_mean"].notna() & out["item_mean"].notna()].copy()
    out["dmv"] = (out["user_mean"] + out["item_mean"]) / 2
    out["eccentricity"] = (out[value_col] - out["dmv"]).abs()
    return out


def eauc(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    pred_col: str = "pred",
    user_col: str = "user_id",
    item_col: str = "item_id",
    value_col: str = "rating",
    method: Method = "ordered",
    tails: Literal[1, 2] = 1,
    bins: int = 100,
) -> float:
    """Compute the EAUC of a model's predictions. Lower is better, range ``[0, 1]``.

    Args:
        train: Train observations (used only to compute per-user and per-item means).
        test: Test observations with the model prediction in ``pred_col``. If ``None``, ``train`` is
            scored instead (and must contain ``pred_col``).
        pred_col, user_col, item_col, value_col: Column names.
        method: How the area is integrated. ``"ordered"`` (the paper's Algorithm 1) sorts examples by
            eccentricity and integrates the error directly. ``"grouped"`` first averages the error of
            examples with identical eccentricity. ``"bins"`` averages the error inside ``bins`` equal-width
            eccentricity bins.
        tails: ``1`` uses ``|r - DMV|`` (the paper's definition). ``2`` keeps the sign of ``r - DMV`` and
            integrates over the two-sided axis.
        bins: Number of bins when ``method="bins"``.
    """
    scored = add_eccentricity(train, train if test is None else test, user_col, item_col, value_col)
    _check_columns(scored, [pred_col], "scored data")

    signed = scored[value_col] - scored["dmv"]
    x_all = scored["eccentricity"] if tails == 1 else signed
    scored = scored.assign(_x=x_all, _err=(scored[value_col] - scored[pred_col]).abs())

    span = scored[value_col].max() - scored[value_col].min()

    if method == "ordered":
        ordered = scored.sort_values("_x")
        xx, yy = ordered["_x"].to_numpy(), ordered["_err"].to_numpy()
    elif method == "grouped":
        grouped = scored.groupby("_x")["_err"].mean()
        xx, yy = grouped.index.to_numpy(), grouped.to_numpy()
    elif method == "bins":
        left = -span if tails == 2 else 0
        edges = np.linspace(left, span, bins)
        per_bin = scored.groupby(pd.cut(scored["_x"], edges), observed=False)["_err"].mean()
        xx = per_bin.index.to_series().apply(lambda interval: interval.mid).to_numpy(dtype=float)
        yy = per_bin.to_numpy()
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'ordered', 'grouped' or 'bins'.")

    keep = ~np.isnan(yy)
    return float(_trapezoid(yy[keep], xx[keep]) / span**2)


def eccentricity_error_curve(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pred_col: str = "pred",
    user_col: str = "user_id",
    item_col: str = "item_id",
    value_col: str = "rating",
    bins: int = 20,
) -> pd.DataFrame:
    """Mean absolute error per eccentricity bin: the *Eccentricity vs. Error* curve of the paper.

    Returns a frame with columns ``eccentricity`` (bin midpoint), ``mae`` and ``count``.
    """
    scored = add_eccentricity(train, test, user_col, item_col, value_col)
    scored["error"] = (scored[value_col] - scored[pred_col]).abs()
    span = scored[value_col].max() - scored[value_col].min()
    edges = np.linspace(0, span, bins + 1)
    grouped = scored.groupby(pd.cut(scored["eccentricity"], edges, include_lowest=True), observed=True)["error"]
    curve = grouped.agg(["mean", "count"]).rename(columns={"mean": "mae", "count": "count"})
    curve.insert(0, "eccentricity", [interval.mid for interval in curve.index])
    return curve.reset_index(drop=True)


def dataset_ks(
    data: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    value_col: str = "rating",
) -> float:
    """Entity-wise Kolmogorov-Smirnov uniformity statistic ``D_KS^D`` of a dataset (paper, Eq. 6).

    For every user and item, computes the KS distance between its observed values and a uniform
    distribution over ``[min(r), max(r)]``, then averages over all users and items together. Higher values
    mean entities concentrate their values, which tempts models into predicting the DMV and induces
    eccentricity bias.
    """
    _check_columns(data, [user_col, item_col, value_col], "data")
    lo, hi = data[value_col].min(), data[value_col].max()
    if hi == lo:
        raise ValueError("All observed values are identical; the uniform reference is undefined.")

    def per_entity(key: str) -> np.ndarray:
        frame = data[[key, value_col]].sort_values([key, value_col])
        cdf = ((frame[value_col] - lo) / (hi - lo)).to_numpy()
        n = frame.groupby(key)[value_col].transform("size").to_numpy()
        # Empirical CDF edges of every sorted value; tied values share the block's outer edges
        rank = frame.groupby(key).cumcount().to_numpy()
        upper = (rank + 1) / n
        lower = rank / n
        vals = frame[value_col].to_numpy()
        keys = frame[key].to_numpy()
        new_block = np.r_[True, (keys[1:] != keys[:-1]) | (vals[1:] != vals[:-1])]
        block_id = np.cumsum(new_block) - 1
        block_last = pd.Series(upper).groupby(block_id).transform("max").to_numpy()
        block_first = pd.Series(lower).groupby(block_id).transform("min").to_numpy()
        d = np.maximum(block_last - cdf, cdf - block_first)
        return pd.Series(d).groupby(keys).max().to_numpy()

    stats = np.concatenate([per_entity(user_col), per_entity(item_col)])
    return float(stats.mean())
