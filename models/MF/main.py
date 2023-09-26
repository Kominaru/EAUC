import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch
from dataset import DyadicRegressionDataModule
from model import CollaborativeFilteringModel
from os import path

DATA_DIR = "data/"
DATASET_NAME = "tripadvisor-london"

EMBEDDING_DIM = 512


# Needs to be in a function for PyTorch Lightning workers to work properly in Windows systems
def train_MF():
    """
    Trains a collaborative filtering model for regression over a dyadic dataset.
    """

    # Load the dyadic dataset using the data module
    data_module = DyadicRegressionDataModule(
        DATA_DIR, batch_size=2**15, num_workers=4, test_size=0.1, dataset_name=DATASET_NAME
    )

    # Initialize the collaborative filtering model
    model = CollaborativeFilteringModel(data_module.num_users, data_module.num_items, EMBEDDING_DIM)

    # Initialize early stopping callback
    # Stops when the validation loss doesn't improve by 1e-4 for 5 epochs
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=1e-4)

    # Checkpoint only the weights that give the best validation RMSE, overwriting existing checkpoints
    if path.exists("models/MF/checkpoints/best-model.ckpt"):
        os.remove("models/MF/checkpoints/best-model.ckpt")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="models/MF/checkpoints/",
        filename="best-model",
        save_top_k=1,
        mode="min",
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, data_module)

    # Load the best model from the checkpoint
    model = model.load_from_checkpoint("models/MF/checkpoints/best-model.ckpt")

    # Save the train samples and predictions
    train_samples_df = pd.DataFrame(columns=["user_id", "item_id", "rating", "pred"])
    train_dataloader = torch.utils.data.DataLoader(
        data_module.train_dataset, batch_size=2**15, shuffle=False, num_workers=4
    )

    predictions = trainer.predict(model, dataloaders=train_dataloader)
    for batch, batch_predictions in zip(train_dataloader, predictions):
        user_ids, item_ids, ratings = batch
        rating_preds = batch_predictions.numpy()

        batch_df = pd.DataFrame(
            {
                "user_id": user_ids.numpy(),
                "item_id": item_ids.numpy(),
                "rating": ratings.numpy(),
                "pred": rating_preds,
            }
        )

        train_samples_df = pd.concat([train_samples_df, batch_df])

    # Save the test samples and predictions
    test_samples_df = pd.DataFrame(columns=["user_id", "item_id", "rating", "pred"])
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
    for batch, batch_predictions in zip(data_module.test_dataloader(), predictions):
        user_ids, item_ids, ratings = batch
        rating_preds = batch_predictions.numpy()

        batch_df = pd.DataFrame(
            {
                "user_id": user_ids.numpy(),
                "item_id": item_ids.numpy(),
                "rating": ratings.numpy(),
                "pred": rating_preds,
            }
        )

        test_samples_df = pd.concat([test_samples_df, batch_df])

    # Save the train and test samples with predictions
    os.makedirs("outputs/MF", exist_ok=True)
    train_samples_df.to_csv("outputs/MF/train_samples.csv", index=False)
    test_samples_df.to_csv("outputs/MF/test_samples_with_predictions.csv", index=False)

    # RMSE
    rmse = ((train_samples_df["rating"] - train_samples_df["pred"]) ** 2).mean() ** 0.5
    print(f"Train RMSE: {rmse:.3f}")
    rmse = ((test_samples_df["rating"] - test_samples_df["pred"]) ** 2).mean() ** 0.5
    print(f"Test RMSE:  {rmse:.3f}")


if __name__ == "__main__":
    train_MF()
