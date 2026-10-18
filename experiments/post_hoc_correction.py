"""Post-training eccentricity-bias corrections of a trained model (paper Section VII, Table IV, Fig. 8).

Reads run ``--run`` of ``outputs/<dataset>/<model>/`` and writes the corrected predictions of each strategy
to ``outputs/<dataset>/<model>_correction_<strategy>/``, ready for ``experiments.eccentricity_analysis``.

Strategies (names as in the paper's Table IV):
    linear           Linear correction fitted on the probe, solved for the observed value.
    linear_balanced  Linear correction + ML-RUS (Clipping).          (``eauc.EccentricityCorrector``)
    linear_sigmoid   Linear correction + ML-RUS (Sigmoid), needs torch.
    random_forest    Random Forest correction.                       (``eauc.EccentricityCorrector``)

Run from the repository root, e.g.:
    python -m experiments.post_hoc_correction --dataset netflix-prize --model MF --run 2
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from eauc import (
    EccentricityCorrector,
    add_eccentricity,
    load_model_outputs,
    make_probe,
    ml_rus,
)

STRATEGIES = ("linear", "linear_balanced", "linear_sigmoid", "random_forest")
COLUMNS = ["user_id", "item_id", "rating", "pred"]
FEATURES = ["pred", "user_mean", "item_mean"]
SOURCE = "from_train"  # 1 for probe rows taken from the train set, 0 for those taken from the test set


def mean_bins(means: pd.Series) -> np.ndarray:
    """Half-point bins of a user or item mean."""
    return (2 * means).astype(int).to_numpy()


def rating_levels(rating_values: pd.Series, train: pd.DataFrame, dataset: str) -> dict:
    """ML-RUS arguments: GDSC1 is continuous, so its values are cut into 4 equal-width levels first."""
    if dataset != "gdsc1":
        return {}
    edges = np.linspace(train["rating"].min(), train["rating"].max(), 5)
    return {"levels": pd.cut(rating_values, bins=edges, labels=False)}


def fit_linear(train: pd.DataFrame, probe: pd.DataFrame):
    """Fit ``pred = a + b*rating + c*user_mean + d*item_mean`` on the probe.

    Returns a function that solves the fitted equation for the rating.
    """
    probe = add_eccentricity(train, probe)
    model = LinearRegression().fit(probe[["rating", "user_mean", "item_mean"]], probe["pred"])
    print(
        f"Linear equation: pred = {model.intercept_:.3f} + {model.coef_[0]:.3f} * rating"
        f" + {model.coef_[1]:.3f} * user_mean + {model.coef_[2]:.3f} * item_mean"
    )

    def correct(df: pd.DataFrame) -> np.ndarray:
        rows = add_eccentricity(train, df)
        corrected = (
            rows["pred"] - model.intercept_ - model.coef_[1] * rows["user_mean"] - model.coef_[2] * rows["item_mean"]
        ) / model.coef_[0]
        return np.clip(corrected, train["rating"].min(), train["rating"].max()).to_numpy()

    return correct


def fit_sigmoid(train: pd.DataFrame, probe: pd.DataFrame, dataset: str, epochs: int = 7500):
    """Fit ``rating = min + (max - min) * sigmoid(w . [pred, user_mean, item_mean] + b)`` on a balanced probe.

    ``probe`` must carry the ``SOURCE`` column (see ``main``): rows taken from the train set are balanced in
    their own bins, separately from the rows taken from the test set.
    """
    import torch

    low, high = train["rating"].min(), train["rating"].max()
    probe = add_eccentricity(train, probe)

    offset = 10 * probe[SOURCE].to_numpy()
    group = (mean_bins(probe["user_mean"]) + offset) * 10 + mean_bins(probe["item_mean"]) + offset
    probe = ml_rus(probe, group, "rating", **rating_levels(probe["rating"], train, dataset))

    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.tensor(probe[FEATURES].values, dtype=torch.float)
    target = torch.tensor(probe["rating"].values.reshape(-1, 1), dtype=torch.float)

    def forward(x):
        return torch.sigmoid(model(x)) * (high - low) + low

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(forward(inputs), target)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.3f}")

    def correct(df: pd.DataFrame) -> np.ndarray:
        rows = add_eccentricity(train, df)
        with torch.no_grad():
            corrected = forward(torch.tensor(rows[FEATURES].values, dtype=torch.float)).numpy().reshape(-1)
        return np.clip(corrected, low, high)

    return correct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="MF")
    parser.add_argument("--run", type=int, default=1, help="Run number of the saved predictions.")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    train, test = load_model_outputs(args.model, args.dataset, args.run, args.output_dir)
    # Pairs whose user or item is unseen in train have no means to correct with, so they are dropped
    test = add_eccentricity(train, test)[COLUMNS]
    print(f"#Ratings: {len(train) + len(test)} | Train: {len(train)} | Test: {len(test)}")

    for strategy in args.strategies:
        print(f"\n== {strategy}")
        train_rest, probe, test_rest = make_probe(train.assign(**{SOURCE: 1}), test.assign(**{SOURCE: 0}))
        print(f"Probe size: {len(probe)}")
        train_rest, test_rest = train_rest[COLUMNS], test_rest[COLUMNS]

        if strategy in ("linear_balanced", "random_forest"):
            corrector = EccentricityCorrector(
                "linear" if strategy == "linear_balanced" else "random_forest",
                n_rating_bins=4 if args.dataset == "gdsc1" else None,
            ).fit(train, probe)
            correct = corrector.predict
        elif strategy == "linear":
            correct = fit_linear(train, probe)
        else:
            correct = fit_sigmoid(train, probe, args.dataset)
        corrected = [correct(df) for df in (train_rest, test_rest)]

        out_dir = os.path.join(args.output_dir, args.dataset, f"{args.model}_correction_{strategy}")
        os.makedirs(out_dir, exist_ok=True)
        for name, df, preds in zip(("train", "test"), (train_rest, test_rest), corrected):
            df.assign(pred=preds).to_csv(os.path.join(out_dir, f"{name}_outputs_{args.run}.csv"), index=False)
        print(f"Saved corrected predictions in {out_dir}")


if __name__ == "__main__":
    main()
