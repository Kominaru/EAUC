"""Post-training eccentricity-bias correction (paper, Section VII).

A trained dyadic regressor ``f`` is left untouched. A small *correction set* (a probe) of held-out
predictions is used to fit a function that maps ``(prediction, user mean, item mean)`` to the observed
value, which undoes part of the model's pull towards the Dyadic Mean Value. The probe can be balanced with
ML-RUS (Multi-Level Random Under-Sampling): inside every (user mean, item mean) bin, every observed value is
under-sampled to the count of the rarest one.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

Strategy = Literal["linear", "random_forest"]


def make_probe(
    train: pd.DataFrame,
    test: pd.DataFrame,
    frac: float = 0.2,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carve a correction set out of held-out predictions.

    A fraction ``frac`` of ``test`` is moved to the probe, together with the same number of ``train`` rows
    (so the probe mixes unseen and seen examples). Both frames need a ``pred`` column and a unique index.

    Returns:
        ``(train_rest, probe, test_rest)``: the corrector should be evaluated on ``test_rest``.
    """
    test_probe = test.sample(frac=frac, random_state=random_state)
    test_rest = test.drop(test_probe.index)
    train_probe = train.sample(n=len(test_probe), random_state=random_state)
    probe = pd.concat([test_probe, train_probe])
    train_rest = train.drop(train_probe.index)
    return train_rest, probe, test_rest


class EccentricityCorrector:
    """Learns a post-hoc correction of a dyadic regressor's predictions.

    Args:
        strategy: ``"linear"`` fits ``value ~ pred + user_mean + item_mean`` by least squares;
            ``"random_forest"`` fits a 100-tree forest (depth 10) on the same three features.
        balance: Apply ML-RUS balancing to the probe before fitting.
        bin_width: Width of the user-mean and item-mean bins used by ML-RUS.
        n_rating_bins: If set, observed values are first cut into this many equal-width bins before
            balancing (useful for continuous targets). ``None`` balances each distinct value.
        random_state: Seed for the under-sampling and the forest.
        user_col, item_col, value_col, pred_col: Column names.
    """

    def __init__(
        self,
        strategy: Strategy = "linear",
        balance: bool = True,
        bin_width: float = 0.5,
        n_rating_bins: int | None = None,
        random_state: int = 0,
        user_col: str = "user_id",
        item_col: str = "item_id",
        value_col: str = "rating",
        pred_col: str = "pred",
    ):
        if strategy not in ("linear", "random_forest"):
            raise ValueError(f"Unknown strategy {strategy!r}; expected 'linear' or 'random_forest'.")
        self.strategy = strategy
        self.balance = balance
        self.bin_width = bin_width
        self.n_rating_bins = n_rating_bins
        self.random_state = random_state
        self.user_col, self.item_col = user_col, item_col
        self.value_col, self.pred_col = value_col, pred_col

    # -- helpers -------------------------------------------------------------------------------------

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["user_mean"] = out[self.user_col].map(self.user_means_)
        out["item_mean"] = out[self.item_col].map(self.item_means_)
        return out

    def _bin(self, means: pd.Series) -> np.ndarray:
        return np.trunc(means.to_numpy() / self.bin_width).astype(int)

    def _balance(self, probe: pd.DataFrame) -> pd.DataFrame:
        """ML-RUS: in each (user mean, item mean) bin, under-sample every value to the rarest one."""
        probe = probe.copy()
        probe["_ui_bin"] = self._bin(probe["user_mean"]) * 10 + self._bin(probe["item_mean"])
        key = self.value_col
        if self.n_rating_bins is not None:
            edges = np.linspace(self.value_min_, self.value_max_, self.n_rating_bins + 1)
            probe["_level"] = pd.cut(probe[self.value_col], bins=edges, labels=False)
            key = "_level"

        parts = []
        for _, ui_group in probe.groupby("_ui_bin", sort=True):
            smallest = max(ui_group[key].value_counts().min(), 1)
            for _, level_group in ui_group.groupby(key, sort=True):
                parts.append(level_group.sample(min(len(level_group), smallest), random_state=self.random_state))
        return pd.concat(parts).drop(columns=["_ui_bin"] + (["_level"] if self.n_rating_bins else []))

    # -- API -----------------------------------------------------------------------------------------

    def fit(self, train: pd.DataFrame, probe: pd.DataFrame) -> "EccentricityCorrector":
        """Fit the correction.

        Args:
            train: Train observations; used for per-user / per-item means and the valid value range.
            probe: Correction set with the model's predictions in ``pred_col`` (see :func:`make_probe`).
        """
        self.user_means_ = train.groupby(self.user_col)[self.value_col].mean()
        self.item_means_ = train.groupby(self.item_col)[self.value_col].mean()
        self.value_min_ = train[self.value_col].min()
        self.value_max_ = train[self.value_col].max()

        probe = self._features(probe)
        probe = probe[probe["user_mean"].notna() & probe["item_mean"].notna()]
        if self.balance:
            probe = self._balance(probe)

        if self.strategy == "linear":
            self.columns_ = [self.pred_col, "user_mean", "item_mean"]
            self.model_ = LinearRegression()
        else:
            self.columns_ = ["user_mean", "item_mean", self.pred_col]
            self.model_ = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=4
            )
        self.model_.fit(probe[self.columns_].to_numpy(), probe[self.value_col].to_numpy())
        self.probe_size_ = len(probe)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Corrected predictions for ``df``, clipped to the observed value range.

        Rows whose user or item was not seen in ``train`` keep their original prediction.
        """
        feats = self._features(df)
        seen = (feats["user_mean"].notna() & feats["item_mean"].notna()).to_numpy()
        corrected = df[self.pred_col].to_numpy(dtype=float).copy()
        if seen.any():
            raw = self.model_.predict(feats.loc[seen, self.columns_].to_numpy())
            corrected[seen] = np.clip(raw, self.value_min_, self.value_max_)
        return corrected

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` whose ``pred`` column holds the corrected predictions."""
        out = df.copy()
        out[self.pred_col] = self.predict(df)
        return out
