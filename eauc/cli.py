"""Command-line interface: ``eauc score | curve | ks | correct``."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .correction import EccentricityCorrector, make_probe
from .metrics import dataset_ks, eauc, eccentricity_error_curve


def _add_columns(p: argparse.ArgumentParser, pred: bool = True) -> None:
    p.add_argument("--user-col", default="user_id", help="User column name (default: user_id).")
    p.add_argument("--item-col", default="item_id", help="Item column name (default: item_id).")
    p.add_argument("--value-col", default="rating", help="Observed value column name (default: rating).")
    if pred:
        p.add_argument("--pred-col", default="pred", help="Prediction column name (default: pred).")


def _cols(a: argparse.Namespace) -> dict:
    return dict(user_col=a.user_col, item_col=a.item_col, value_col=a.value_col)


def _report(name: str, test: pd.DataFrame, train: pd.DataFrame, a: argparse.Namespace) -> None:
    err = test[a.value_col] - test[a.pred_col]
    score = eauc(train, test, pred_col=a.pred_col, method=a.method, tails=a.tails, bins=a.bins, **_cols(a))
    print(f"{name:<10} EAUC={score:.4f}  RMSE={np.sqrt((err**2).mean()):.4f}  MAE={err.abs().mean():.4f}")


def _score(a: argparse.Namespace) -> None:
    _report("model", pd.read_csv(a.test), pd.read_csv(a.train), a)


def _curve(a: argparse.Namespace) -> None:
    curve = eccentricity_error_curve(
        pd.read_csv(a.train), pd.read_csv(a.test), pred_col=a.pred_col, bins=a.bins, **_cols(a)
    )
    if a.out:
        curve.to_csv(a.out, index=False)
    else:
        print(curve.to_string(index=False))
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(curve["eccentricity"], curve["mae"])
        ax.set_xlabel("Eccentricity")
        ax.set_ylabel("Mean absolute error")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=200)


def _ks(a: argparse.Namespace) -> None:
    print(f"D_KS = {dataset_ks(pd.read_csv(a.data), **_cols(a)):.4f}")


def _correct(a: argparse.Namespace) -> None:
    train, test = pd.read_csv(a.train), pd.read_csv(a.test)
    train_rest, probe, test_rest = make_probe(train, test, frac=a.probe_frac, random_state=a.seed)
    corrector = EccentricityCorrector(
        strategy=a.strategy, balance=not a.no_balance, random_state=a.seed, pred_col=a.pred_col, **_cols(a)
    ).fit(train, probe)
    corrected = corrector.transform(test_rest)
    _report("before", test_rest, train, a)
    _report("after", corrected, train, a)
    corrected.to_csv(a.out, index=False)
    print(f"Saved {len(corrected)} corrected test rows to {a.out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eauc", description="Measure and mitigate eccentricity bias.")
    sub = parser.add_subparsers(dest="command", required=True)

    def metric_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--method", choices=["ordered", "grouped", "bins"], default="ordered")
        p.add_argument("--tails", type=int, choices=[1, 2], default=1)
        p.add_argument("--bins", type=int, default=100, help="Bins for --method bins.")

    p = sub.add_parser("score", help="EAUC, RMSE and MAE of a model's test predictions.")
    p.add_argument("--train", required=True, help="CSV with the train observations.")
    p.add_argument("--test", required=True, help="CSV with test observations and predictions.")
    _add_columns(p)
    metric_args(p)
    p.set_defaults(func=_score)

    p = sub.add_parser("curve", help="Eccentricity vs. error curve.")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--out", help="Write the curve to this CSV instead of printing it.")
    p.add_argument("--plot", help="Also save a PNG plot (needs matplotlib).")
    _add_columns(p)
    p.set_defaults(func=_curve)

    p = sub.add_parser("ks", help="Dataset-level entity-wise KS uniformity statistic (D_KS).")
    p.add_argument("--data", required=True, help="CSV with all observations.")
    _add_columns(p, pred=False)
    p.set_defaults(func=_ks)

    p = sub.add_parser("correct", help="Fit a post-hoc correction and report EAUC before and after.")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True, help="CSV for the corrected test predictions.")
    p.add_argument("--strategy", choices=["linear", "random_forest"], default="linear")
    p.add_argument("--no-balance", action="store_true", help="Disable ML-RUS balancing of the probe.")
    p.add_argument("--probe-frac", type=float, default=0.2, help="Fraction of test moved to the probe.")
    p.add_argument("--seed", type=int, default=0)
    _add_columns(p)
    metric_args(p)
    p.set_defaults(func=_correct)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
