"""Trains the MF baseline on a dyadic dataset and saves its predictions.

Run from the repository root, e.g. ``python models/MF/main.py --dataset ml-1m``. Predictions are written to
``outputs/<dataset>/MF/{train,test}_outputs_<n>.csv``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

sys.path.insert(0, os.path.dirname(__file__))
from dataset import DyadicRegressionDataModule  # noqa: E402
from model import CollaborativeFilteringModel  # noqa: E402

from eauc import save_model_outputs  # noqa: E402

logging.getLogger("lightning").setLevel(0)

CHECKPOINT_DIR = "checkpoints/MF"

# Hyperparameters of the paper (Table V): embedding dimension, learning rate and L2 weight regularization.
PAPER_HPARAMS = {
    "ml-1m": (512, 1e-3, 1e-5),
    "netflix-prize": (512, 1e-3, 1e-5),
    "gdsc1": (512, 1e-3, 1e-5),
    "ctrpv2": (512, 1e-3, 1e-5),
    "dot_2023": (8, 1e-3, 1e-4),
    "kiva-ml-17": (128, 5e-4, 1e-4),
}
DEFAULT_HPARAMS = (256, 1e-3, 1e-3)


def predict_dataframe(trainer: pl.Trainer, model: pl.LightningModule, dataloader) -> pd.DataFrame:
    """Predictions of ``model`` over ``dataloader`` as a ``user_id, item_id, rating, pred`` frame."""
    predictions = trainer.predict(model, dataloaders=dataloader)
    frames = [
        pd.DataFrame(
            {
                "user_id": user_ids.numpy(),
                "item_id": item_ids.numpy(),
                "rating": ratings.numpy(),
                "pred": batch_predictions.numpy(),
            }
        )
        for (user_ids, item_ids, ratings), batch_predictions in zip(dataloader, predictions)
    ]
    return pd.concat(frames, ignore_index=True)


def rmse(df: pd.DataFrame) -> float:
    return float(((df["rating"] - df["pred"]) ** 2).mean() ** 0.5)


# Kept in a function so that PyTorch Lightning workers also work on Windows
def train_mf(
    dataset_name: str,
    embedding_dim: int,
    learning_rate: float,
    l2_reg: float,
    data_dir: str = "data",
    max_epochs: int = 1000,
    batch_size: int = 2**10,
    num_workers: int = 4,
    seed: int | None = None,
    save_outputs: bool = True,
) -> None:
    data_module = DyadicRegressionDataModule(
        data_dir, dataset_name, batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    model = CollaborativeFilteringModel(
        int(data_module.num_users),
        int(data_module.num_items),
        embedding_dim=embedding_dim,
        l2_reg=l2_reg,
        lr=learning_rate,
        rating_range=(float(data_module.min_rating), float(data_module.max_rating)),
    )

    # Stop when the validation loss does not improve by 1e-4 for 20 epochs; keep the best weights
    best_checkpoint = os.path.join(CHECKPOINT_DIR, "best-model.ckpt")
    if os.path.exists(best_checkpoint):
        os.remove(best_checkpoint)
    checkpoint = ModelCheckpoint(
        monitor="val_loss", dirpath=CHECKPOINT_DIR, filename="best-model", save_top_k=1, mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[EarlyStopping(monitor="val_loss", patience=20, mode="min", min_delta=1e-4), checkpoint],
    )
    trainer.fit(model, data_module)
    model = CollaborativeFilteringModel.load_from_checkpoint(best_checkpoint)

    train_loader = torch.utils.data.DataLoader(
        data_module.train_dataset,
        batch_size=2**15,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    train_df = predict_dataframe(trainer, model, train_loader)
    test_df = predict_dataframe(trainer, model, data_module.test_dataloader())

    if save_outputs:
        save_model_outputs(
            train_df,
            test_df,
            "MF",
            dataset_name,
            {"embedding_dim": embedding_dim, "l2_reg": l2_reg, "learning_rate": learning_rate},
        )
    print(f"Train RMSE: {rmse(train_df):.3f}")
    print(f"Test RMSE:  {rmse(test_df):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--embedding-dim", type=int, help="Defaults to the paper's value for the dataset.")
    parser.add_argument("--learning-rate", type=float, help="Defaults to the paper's value for the dataset.")
    parser.add_argument("--l2-reg", type=float, help="Defaults to the paper's value for the dataset.")
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2**10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, help="Seed of the random train-test split.")
    parser.add_argument("--no-save", action="store_true", help="Do not write the predictions to outputs/.")
    args = parser.parse_args()

    dim, lr, l2 = PAPER_HPARAMS.get(args.dataset, DEFAULT_HPARAMS)
    train_mf(
        args.dataset,
        embedding_dim=args.embedding_dim or dim,
        learning_rate=args.learning_rate or lr,
        l2_reg=args.l2_reg if args.l2_reg is not None else l2,
        data_dir=args.data_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        save_outputs=not args.no_save,
    )


if __name__ == "__main__":
    main()
