"""End-to-end EAUC example on synthetic data: no downloads, runs in seconds.

A dyadic dataset is simulated and scored with three predictors:
  * Random:      uniform predictions.
  * Dyad average: always predicts the Dyadic Mean Value (the extreme of eccentricity bias).
  * Biased model: stands in for a model that mostly learned entity means (shrunk towards the DMV).
Then a post-hoc correction is fitted on a probe and EAUC is re-measured.

Run:  python examples/synthetic_example.py [--plot assets/eccentricity_curve.png]
"""

import argparse

import numpy as np
import pandas as pd

from eauc import EccentricityCorrector, add_eccentricity, eauc, eccentricity_error_curve, make_probe


def make_data(n_users=600, n_items=400, n_obs=60_000, seed=0):
    rng = np.random.default_rng(seed)
    user_bias, item_bias = rng.normal(0, 0.6, n_users), rng.normal(0, 0.6, n_items)
    users, items = rng.integers(0, n_users, n_obs), rng.integers(0, n_items, n_obs)
    affinity = rng.normal(0, 1.0, n_obs)
    rating = np.clip(np.round(3.3 + user_bias[users] + item_bias[items] + affinity), 1, 5)
    df = pd.DataFrame({"user_id": users, "item_id": items, "rating": rating})
    df = df.drop_duplicates(["user_id", "item_id"]).reset_index(drop=True)
    cut = int(0.9 * len(df))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", help="Save the eccentricity vs. error plot to this path (needs matplotlib).")
    args = parser.parse_args()

    train, test = make_data()
    rng = np.random.default_rng(1)

    def with_predictions(df):
        """Attach the three predictors' outputs to every row whose user and item appear in train."""
        scored = add_eccentricity(train, df)
        dmv = scored["dmv"]
        shrunk = dmv + 0.3 * (scored["rating"] - dmv) + rng.normal(0, 0.25, len(scored))
        return scored[["user_id", "item_id", "rating"]].assign(
            random=rng.uniform(1, 5, len(scored)), dyad_average=dmv, biased=np.clip(shrunk, 1, 5)
        )

    train_pred, test_pred = with_predictions(train), with_predictions(test)
    results = {
        "random": test_pred.assign(pred=test_pred["random"]),
        "dyad average": test_pred.assign(pred=test_pred["dyad_average"]),
        "biased model": test_pred.assign(pred=test_pred["biased"]),
    }

    _, probe, test_rest = make_probe(train_pred.assign(pred=train_pred["biased"]), results["biased model"])
    corrector = EccentricityCorrector("linear").fit(train, probe)
    results["biased model + correction"] = corrector.transform(test_rest)

    print(f"{'predictor':<28}{'RMSE':>8}{'EAUC':>8}")
    for name, df in results.items():
        rmse = float(np.sqrt(((df["rating"] - df["pred"]) ** 2).mean()))
        print(f"{name:<28}{rmse:>8.3f}{eauc(train, df):>8.3f}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.5, 5))
        for name, df in results.items():
            curve = eccentricity_error_curve(train, df, bins=12)
            ax.plot(curve["eccentricity"], curve["mae"], marker="o", ms=3, label=name)
        ax.set_xlabel("Eccentricity  $|r_{ui} - DMV_{ui}|$")
        ax.set_ylabel("Mean absolute error")
        ax.set_title("Eccentricity vs. error (synthetic data)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
