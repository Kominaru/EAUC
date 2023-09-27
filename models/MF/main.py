import os
import random
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from dataset import DyadicRegressionDataModule
from model import CollaborativeFilteringModel
from os import path


# Needs to be in a function for PyTorch Lightning workers to work properly in Windows systems
def train_MF(
    dataset_name="netflix-prize",
    embedding_dim=8,
    data_dir="data",
    max_epochs=1000,
    batch_size=2**14,
    num_workers=4,
    l2_reg=1e-5,
    learning_rate=5e-4,
):
    """
    Trains a collaborative filtering model for regression over a dyadic dataset.
    """

    # Load the dyadic dataset using the data module
    data_module = DyadicRegressionDataModule(
        data_dir, batch_size=batch_size, num_workers=num_workers, test_size=0.1, dataset_name=dataset_name
    )

    # Initialize the collaborative filtering model
    model = CollaborativeFilteringModel(
        data_module.num_users, data_module.num_items, embedding_dim=embedding_dim, l2_reg=l2_reg, lr=learning_rate
    )

    # Initialize early stopping callback
    # Stops when the validation loss doesn't improve by 1e-4 for 10 epochs
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min", min_delta=1e-4)

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
        max_epochs=max_epochs,
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

    return rmse


if __name__ == "__main__":
    MODE = "train"

    if MODE == "train":
        train_MF()

    elif MODE == "tune":
        embedding_dims = [8, 32, 128, 512, 1024]
        l2_regs = [0, 1e-5, 1e-4, 1e-3, 1e-2]
        learning_rates = [1e-5, 1e-4, 5e-4, 1e-3]

        # Choose n random hyperparameter combinations
        NUM_TRIALS = 50

        # Create a dataframe to store the results
        results_df = []

        for i in range(NUM_TRIALS):
            embedding_dim = random.choice(embedding_dims)
            l2_reg = random.choice(l2_regs)
            learning_rate = random.choice(learning_rates)

            rmse = train_MF(embedding_dim=embedding_dim, l2_reg=l2_reg, learning_rate=learning_rate)

            results_df.append([embedding_dim, l2_reg, learning_rate, rmse])

            print("==============================================")
            print(f"Trial {i+1}/{NUM_TRIALS} completed")
            print(f"Hyperparameters: embedding_dim={embedding_dim}, l2_reg={l2_reg}, learning_rate={learning_rate}")
            print(f"RMSE: {rmse:.3f}")
            print("==============================================")

        results_df = pd.DataFrame(results_df, columns=["embedding_dim", "l2_reg", "learning_rate", "rmse"])
        results_df.to_csv("outputs/MF/tune_results.csv", index=False)
