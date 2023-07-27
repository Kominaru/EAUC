import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch
from dataset import MovieLensDataModule
from model import CollaborativeFilteringModel
from os import path
def main():
    # Specify the data directory and other hyperparameters
    data_dir = 'data/'  # Update this with the correct directory path
    embedding_dim = 512

    # Load the MovieLens dataset using the data module
    data_module = MovieLensDataModule(data_dir, batch_size=2**15, num_workers=4, test_size=0.1)

    # Save the train samples DataFrame

    # Initialize the collaborative filtering model
    model = CollaborativeFilteringModel(data_module.num_users, data_module.num_movies,
                                        embedding_dim)

    # Initialize early stopping callback
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.0001)

    # Checkpoint only the weights that give the best validation RMSE, overwriting existing checkpoints

    if path.exists('models/MF/checkpoints/best-model.ckpt'):
        os.remove('models/MF/checkpoints/best-model.ckpt')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='models/MF/checkpoints/',
        filename='best-model',
        save_top_k=1,
        mode='min',

    )

    # Initialize CSV logger to save training logs
    csv_logger = CSVLogger(save_dir='models/MF/logs/', name='collaborative_filtering')

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=1000, accelerator='auto', callbacks=[early_stop_callback, checkpoint_callback], logger=csv_logger)

    # Train the model
    trainer.fit(model, data_module)

    

    # Save the test samples with predictions DataFrame
    train_samples_df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'pred'])
    test_samples_df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'pred'])

    

    # Load the best model from the checkpoint
    model = model.load_from_checkpoint('models/MF/checkpoints/best-model.ckpt')

    # Recreate the train dataloader with shuffle=False
    train_dataloader = torch.utils.data.DataLoader(data_module.train_dataset, batch_size=2**15, shuffle=False, num_workers=4)

    # Use the predict method with the train dataloader
    predictions = trainer.predict(model, dataloaders=train_dataloader)
    for batch, batch_predictions in zip(train_dataloader, predictions):
        user_ids, movie_ids, ratings = batch
        rating_preds = batch_predictions.numpy()

        batch_df = pd.DataFrame({
            'user_id': user_ids.numpy(),
            'movie_id': movie_ids.numpy(),
            'rating': ratings.numpy(),
            'pred': rating_preds
        })

        train_samples_df = pd.concat([train_samples_df, batch_df])

    # Use the predict method with the test dataloader
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
    for batch, batch_predictions in zip(data_module.test_dataloader(), predictions):
        user_ids, movie_ids, ratings = batch
        rating_preds = batch_predictions.numpy()

        batch_df = pd.DataFrame({
            'user_id': user_ids.numpy(),
            'movie_id': movie_ids.numpy(),
            'rating': ratings.numpy(),
            'pred': rating_preds
        })

        test_samples_df = pd.concat([test_samples_df, batch_df])

    train_tuples = train_samples_df[['user_id', 'movie_id']].apply(tuple, axis=1)
    test_tuples = test_samples_df[['user_id', 'movie_id']].apply(tuple, axis=1)

    print('Repeated samples: ', test_tuples.isin(train_tuples).sum())

    # Compute the RMSE on the train set
    rmse = ((train_samples_df['rating'] - train_samples_df['pred']) ** 2).mean() ** 0.5
    print(f'Train RMSE: {rmse:.3f}')

    # Compute the RMSE on the test set
    rmse = ((test_samples_df['rating'] - test_samples_df['pred']) ** 2).mean() ** 0.5
    print(f'Test RMSE: {rmse:.3f}')



    # Save the train and test samples with predictions DataFrame
    train_samples_df.to_csv('outputs/MF/train_samples.csv', index=False)
    test_samples_df.to_csv('outputs/MF/test_samples_with_predictions.csv', index=False)

if __name__ == "__main__":
    main()