"""Read and write the per-run prediction files used by the experiment scripts.

Each run of a model on a dataset is stored as ``<root>/<dataset>/<model>/train_outputs_<n>.csv`` and
``test_outputs_<n>.csv`` with columns ``user_id, item_id, rating, pred``.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

_RUN_FILE = re.compile(r"(?:train|test)_outputs_(\d+)\.csv$")


def save_model_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    model_params: dict | None = None,
    root: str | Path = "outputs",
) -> int:
    """Save the predictions of a run under the next free run number and log its parameters.

    Returns:
        The run number used in the file names.
    """
    folder = Path(root) / dataset_name / model_name
    folder.mkdir(parents=True, exist_ok=True)

    taken = [int(m.group(1)) for f in folder.iterdir() if (m := _RUN_FILE.match(f.name))]
    run = max(taken, default=0) + 1

    train_df.to_csv(folder / f"train_outputs_{run}.csv", index=False)
    test_df.to_csv(folder / f"test_outputs_{run}.csv", index=False)

    fields = [str(run), str(datetime.now())] + [f"{k}: {v}" for k, v in (model_params or {}).items()]
    with open(folder / "params.txt", "a") as f:
        f.write("\t".join(fields) + "\n")
    return run


def load_model_outputs(
    model_name: str,
    dataset_name: str,
    run: int,
    root: str | Path = "outputs",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the ``(train, test)`` predictions of one run."""
    folder = Path(root) / dataset_name / model_name
    return pd.read_csv(folder / f"train_outputs_{run}.csv"), pd.read_csv(folder / f"test_outputs_{run}.csv")
