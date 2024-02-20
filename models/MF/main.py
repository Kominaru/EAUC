import os

# Disable pytorch lightning warnings

import random
import pandas as pd
import logging

logging.getLogger("lightning").setLevel(0)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from dataset import DyadicRegressionDataModule
from model import CollaborativeFilteringModel, CrossAttentionMFModel
from os import path
from save_model_outputs import save_model_outputs


MODEL = "CrossAttMF"


# Needs to be in a function for PyTorch Lightning workers to work properly in Windows systems
def train_MF(
    dataset_name="ml-1m",
    embedding_dim=256,  # 128 for tripadvisor-london and ml-100k, 8 for douban-monti, 512 for the rest
    data_dir="data",
    max_epochs=1000,
    batch_size=2**15,
    num_workers=4,
    l2_reg=1e-3,  # 1e-4 for tripadvisor-london and ml-100k
    learning_rate=1e-4,  # 5e-4 for ml-100k
    dropout=0.0,
    verbose=0,
    tune=False,
):
    """
    Trains a collaborative filtering model for regression over a dyadic dataset .
    """

    if tune:
        l2_reg = 10**l2_reg
        learning_rate = 10**learning_rate
        embedding_dim = int(2**embedding_dim)

    # Load the dyadic dataset using the data module
    data_module = DyadicRegressionDataModule(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        test_size=0.1,
        dataset_name=dataset_name,
        verbose=verbose,
    )

    # Initialize the collaborative filtering model
    if MODEL == "MF":
        model = CollaborativeFilteringModel(
            data_module.num_users,
            data_module.num_items,
            embedding_dim=embedding_dim,
            l2_reg=l2_reg,
            lr=learning_rate,
            dropout=dropout,
            rating_range=(data_module.min_rating, data_module.max_rating),
        )
    elif MODEL == "CrossAttMF":
        model = CrossAttentionMFModel(
            data_module.num_users,
            data_module.num_items,
            usr_avg=data_module.avg_user_rating,
            item_avg=data_module.avg_item_rating,
            embedding_dim=embedding_dim,
            l2_reg=l2_reg,
            lr=learning_rate,
            dropout=dropout,
            rating_range=(data_module.min_rating, data_module.max_rating),
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
        enable_model_summary=verbose,
        enable_progress_bar=verbose,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Load the best model from the checkpoint
    model = model.load_from_checkpoint("models/MF/checkpoints/best-model.ckpt")

    train_dataloader = torch.utils.data.DataLoader(
        data_module.train_dataset, batch_size=2**15, shuffle=False, num_workers=4, persistent_workers=True
    )

    train_samples_data = []
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

        train_samples_data.append(batch_df)

    train_samples_df = pd.concat(train_samples_data, ignore_index=True)

    # Save the test samples and predictions
    test_samples_data = []
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

        test_samples_data.append(batch_df)

    test_samples_df = pd.concat(test_samples_data, ignore_index=True)

    # Save the train and test samples with predictions
    save_model_outputs(
        train_samples_df,
        test_samples_df,
        "MF",
        dataset_name,
        {"embedding_dim": embedding_dim, "l2_reg": l2_reg, "learning_rate": learning_rate, "dropout": dropout},
    )

    # RMSE
    rmse = ((train_samples_df["rating"] - train_samples_df["pred"]) ** 2).mean() ** 0.5
    print(f"Train RMSE: {rmse:.3f}")
    rmse = ((test_samples_df["rating"] - test_samples_df["pred"]) ** 2).mean() ** 0.5
    print(f"Test RMSE:  {rmse:.3f}")

    # Append the hyperparameters and RMSE to a file
    # Create the file if it doesn't exist
    # if not os.path.exists(f"outputs/{dataset_name}/MF/resultstune.txt"):
    #     with open(f"outputs/{dataset_name}/MF/resultstune.txt", "w") as f:
    #         f.write("")
    #         f.close()
    # with open(f"outputs/{dataset_name}/MF/resultstune.txt", "a") as f:
    #     f.write(
    #         f"embedding_dim={embedding_dim}, l2_reg={l2_reg}, learning_rate={learning_rate}, dropout={dropout} rmse={rmse}\n"
    #     )

    return -rmse


if __name__ == "__main__":
    MODE = "tune"

    if MODE == "train":
        train_MF(verbose=1)

    elif MODE == "tune":

        # Bayesian optimization
        from bayes_opt import BayesianOptimization

        # Bounded region of parameter space
        pbounds = {
            "embedding_dim": (3, 10),
            "l2_reg": (-6, -2),
            "learning_rate": (-5, -2),
            "dropout": (0, 0.5),
        }

        def train_MF_tune(embedding_dim, l2_reg, learning_rate, dropout):
            return train_MF(
                embedding_dim=embedding_dim, l2_reg=l2_reg, learning_rate=learning_rate, dropout=dropout, tune=True
            )

        optimizer = BayesianOptimization(
            f=train_MF_tune,
            pbounds=pbounds,
        )

        optimizer.maximize(init_points=10, n_iter=30)

        print(optimizer.max)
