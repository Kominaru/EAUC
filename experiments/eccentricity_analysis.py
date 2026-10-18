"""Compare models by RMSE, MAE and EAUC and plot their eccentricity vs. error curves (paper Figs. 2, 3, 5, 9).

Reads the runs saved by the model scripts (``outputs/<dataset>/<model>/{train,test}_outputs_<n>.csv``).
The pseudo-models ``Random`` (uniform predictions) and ``average`` (the Dyadic Mean Value) are computed on
the fly, and corrected variants are listed as ``<model>_correction_<strategy>``.

Run from the repository root, e.g.:
    python -m experiments.eccentricity_analysis --models MF GCMC GLOCAL_K average Random --dataset ml-1m \
        --first-run 1 --n-runs 5
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eauc import add_eccentricity, eauc, load_model_outputs

OUTPUT_DIR = "outputs"

DATASET_NAMES = {
    "ml-1m": "MovieLens 1M",
    "ml-100k": "MovieLens 100K",
    "ml-10m": "MovieLens 10M",
    "douban-monti": "Douban Monti",
    "tripadvisor-london": "TripAdvisor",
    "kiva-ml-17": "Kiva Microloans 2017",
    "gdsc1": "GDSC1",
    "ctrpv2": "CTRPv2",
    "dot_2023": "IMF DOTS 2023",
    "netflix-prize": "Netflix Prize",
}

# model -> (display name, colour, line style)
STYLES = {
    "MF": ("MF", "#8f4de1", "-"),
    "GLOCAL_K": ("GLOCAL-K", "#60c138", "-"),
    "GCMC": ("GC-MC", "#dd3dbb", "-"),
    "BAYESIAN_SVD++": ("Bayesian SVD++", "#daa91f", "-"),
    "Random": ("Random", "gray", "-"),
    "average": ("Dyadic Average", "black", "-"),
    "MF_correction_linear": ("MF w/ Linear corr.", "#8f4de1", "--"),
    "MF_correction_linear_sigmoid": ("MF w/ Linear corr.\n+ ML-RUS (Sigmoid)", "#8f4de1", "-."),
    "MF_correction_linear_balanced": ("MF w/ Linear corr.\n+ ML-RUS (Clipping)", "#8f4de1", ":"),
    "MF_correction_random_forest": ("MF w/ Random Forest corr.", "#8f4de1", (0, (1, 10))),
}

Run = tuple[pd.DataFrame, pd.DataFrame]  # (train, test)


def random_baseline(runs: list[Run], rng: np.random.Generator) -> list[Run]:
    """Uniform random predictions over the observed value range."""
    out = []
    for train, test in runs:
        low, high = train["rating"].min(), train["rating"].max()
        out.append(
            (
                train.assign(pred=rng.uniform(low, high, len(train))),
                test.assign(pred=rng.uniform(low, high, len(test))),
            )
        )
    return out


def dyad_average_baseline(runs: list[Run]) -> list[Run]:
    """Predict the Dyadic Mean Value of each pair (the global train mean for unseen users or items)."""
    out = []
    for train, test in runs:
        user_mean = train.groupby("user_id")["rating"].mean()
        item_mean = train.groupby("item_id")["rating"].mean()
        train_pred = (train["user_id"].map(user_mean) + train["item_id"].map(item_mean)) / 2
        test_pred = (test["user_id"].map(user_mean) + test["item_id"].map(item_mean)) / 2
        out.append((train.assign(pred=train_pred), test.assign(pred=test_pred.fillna(train["rating"].mean()))))
    return out


def print_tables(runs_by_model: dict[str, list[Run]], method: str, tails: int, bins: int) -> None:
    """Print RMSE, MAE and EAUC for the train and test sets of every run of every model."""
    metrics = {
        "RMSE": lambda tr, te: tuple(((d["rating"] - d["pred"]) ** 2).mean() ** 0.5 for d in (tr, te)),
        "MAE": lambda tr, te: tuple((d["rating"] - d["pred"]).abs().mean() for d in (tr, te)),
        "EAUC": lambda tr, te: (
            eauc(tr, method=method, tails=tails, bins=bins),
            eauc(tr, te, method=method, tails=tails, bins=bins),
        ),
    }
    for name, metric in metrics.items():
        print(f"\n{name}\n{'-' * 34}\n{'Model':<20}{'Train':<8}Test")
        for model, runs in runs_by_model.items():
            for train, test in runs:
                train_value, test_value = metric(train, test)
                print(f"{model:<20}{train_value:<8.3f}{test_value:.3f}")


def plot_error_by_eccentricity(
    runs_by_model: dict[str, list[Run]], split: str, dataset: str, tails: int, num_bins: int, out_dir: str
) -> None:
    """Mean absolute error per eccentricity bin, averaged over runs (shaded: standard deviation).

    ``split`` is ``"train"`` (the train set scored against its own means) or ``"test"``.
    """
    span = max(df["rating"].max() - df["rating"].min() for runs in runs_by_model.values() for run in runs for df in run)
    edges = np.linspace(0 if tails == 1 else -span, span, num_bins)

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.rcParams.update({"font.size": 16})
    x_max = 0.0
    for model, runs in runs_by_model.items():
        per_run = []
        for train, test in runs:
            scored = add_eccentricity(train, train if split == "train" else test)
            x = scored["eccentricity"] if tails == 1 else scored["rating"] - scored["dmv"]
            error = (scored["rating"] - scored["pred"]).abs()
            per_bin = error.groupby(pd.cut(x, edges), observed=True).mean().dropna()
            per_run.append(pd.Series(per_bin.to_numpy(), index=[b.mid for b in per_bin.index]))

        stats = pd.concat(per_run).groupby(level=0).agg(["mean", "std"]).dropna(subset=["mean"])
        name, color, style = STYLES[model]
        ax.plot(stats.index, stats["mean"], label=name, color=color, linestyle=style)
        ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=0.1, color=color)
        x_max = max(x_max, stats.index.max())

    ax.plot([0, span], [0, span], color="black", alpha=0.2)  # y = x
    ax.set(xlim=(0, x_max), ylim=(0, x_max), title=DATASET_NAMES.get(dataset, dataset))
    ax.set_xlabel("Rating Eccentricity $Ecc_{ui}$")
    ax.set_ylabel(r"Prediction Error $|r_{ui} - \hat{r}_{ui}|$")
    ax.grid(alpha=0.3)
    if "MF_correction_linear" in runs_by_model or dataset == "ml-1m":
        ax.legend(loc="upper left", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, f"eccentricity_error_{split}_{dataset}.pdf")
    fig.savefig(path)
    print(f"Saved {path}")


def plot_ratings_vs_predictions(train: pd.DataFrame, test: pd.DataFrame, dmv_ranges, out_dir: str) -> None:
    """Observed vs. predicted value (mean and std) for pairs whose DMV falls in each range (paper Fig. 3)."""
    scored = add_eccentricity(train, test)
    low, high = train["rating"].min(), train["rating"].max()

    fig, ax = plt.subplots(figsize=(6.25, 5))
    for dmv_range, style in zip(dmv_ranges, ["-.", ":"]):
        subset = scored[scored["dmv"].between(*dmv_range)]
        stats = subset.groupby("rating")["pred"].agg(["mean", "std"])
        label = f"$(u, i)$ pairs with\n${dmv_range[0]} \\leq DMV_{{ui}} \\leq {dmv_range[1]}$"
        ax.plot(stats.index, stats["mean"], color="red", linestyle=style, label=label)
        ax.fill_between(
            stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=0.05, color="red"
        )

    ax.plot([low, high], [low, high], color="black", linestyle="--", label="Ideal regressor\nfor any $DMV_{ui}$")
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="Rating $r_{ui}$", ylabel=r"Prediction $\hat{r}_{ui}$")
    ax.set_title("Test ratings vs. predictions\ndepending on $DMV_{ui}$")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    path = os.path.join(out_dir, "ratings_vs_predictions.pdf")
    fig.savefig(path)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="Models to compare (see STYLES).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--first-run", type=int, default=1, help="Number of the first run to load.")
    parser.add_argument("--n-runs", type=int, default=1, help="Number of consecutive runs to load.")
    parser.add_argument("--method", choices=["ordered", "grouped", "bins"], default="ordered", help="EAUC integration.")
    parser.add_argument("--tails", type=int, choices=[1, 2], default=1)
    parser.add_argument("--bins", type=int, default=20, help="Bins of the EAUC 'bins' method.")
    parser.add_argument("--plot-bins", type=int, default=25, help="Bins of the plotted curves.")
    parser.add_argument(
        "--dmv-ranges",
        type=float,
        nargs=4,
        metavar=("LO1", "HI1", "LO2", "HI2"),
        help="Also plot observed vs. predicted values for two DMV ranges (Fig. 3).",
    )
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng()
    real_models = [m for m in args.models if m not in ("Random", "average")]
    if not real_models:
        parser.error("Provide at least one trained model: Random and average are derived from its runs.")

    runs_by_model = {
        model: [
            load_model_outputs(model, args.dataset, run, OUTPUT_DIR)
            for run in range(args.first_run, args.first_run + args.n_runs)
        ]
        for model in real_models
    }
    template = runs_by_model[real_models[0]]
    if "Random" in args.models:
        runs_by_model["Random"] = random_baseline(template, rng)
    if "average" in args.models:
        runs_by_model["average"] = dyad_average_baseline(template)
    runs_by_model = {model: runs_by_model[model] for model in args.models}

    print_tables(runs_by_model, args.method, args.tails, args.bins)

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ("train", "test"):
        plot_error_by_eccentricity(runs_by_model, split, args.dataset, args.tails, args.plot_bins, args.out_dir)
    if args.dmv_ranges:
        lo1, hi1, lo2, hi2 = args.dmv_ranges
        train, test = runs_by_model[args.models[0]][0]
        plot_ratings_vs_predictions(train, test, [(lo1, hi1), (lo2, hi2)], args.out_dir)


if __name__ == "__main__":
    main()
