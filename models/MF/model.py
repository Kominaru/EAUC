from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

class CollaborativeFilteringModel(LightningModule):
    '''
    Collaborative filtering model that predicts the rating of a movie for a user
    Ratings are computed as:
        r_hat = dot(user_embedding, movie_embedding) + user_bias + movie_bias + global_bias
    
    '''
    def __init__(self, num_users, num_movies, embedding_dim = 100, lr=5e-4, l2_reg=1e-5):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.global_bias = nn.Parameter(torch.tensor(0).float())

        self.lr = lr
        self.l2_reg = l2_reg

        self.loss_fn = nn.MSELoss()

        # Initialize embedding weights with Xavier distribution
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        
        # Initialize bias weights to zero
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

        # Metrics
        self.rmse = torchmetrics.MeanSquaredError(squared=False)

        # Clip the ratings between 1.0 and 5.0 for predictions
        self.clamp_ratings = lambda x: torch.clamp(x, min=1.0, max=5.0)

        # Save hyperparameters
        self.save_hyperparameters()
        

    def forward(self, user_ids, movie_ids):

        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)

        dot_product = torch.sum(user_embeds * movie_embeds, dim=1, keepdim=True)

        user_bias = self.user_bias(user_ids)
        movie_bias = self.movie_bias(movie_ids)

        prediction = dot_product + user_bias + movie_bias + self.global_bias

        return prediction

    def training_step(self, batch, batch_idx):
        user_ids, movie_ids, ratings = batch
        rating_pred = self(user_ids, movie_ids).squeeze()

        # Compute Mean Squared Error (MSE) loss
        loss = nn.MSELoss()(rating_pred, ratings)

        # Logging the loss value
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, movie_ids, ratings = batch
        rating_pred = self(user_ids, movie_ids).squeeze()

        loss = nn.MSELoss()(rating_pred, ratings)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Clamp the predicted ratings for RMSE computation
        rating_pred_clamped = self.clamp_ratings(rating_pred)

        # Compute RMSE using torchmetrics
        self.rmse.update(rating_pred_clamped, ratings)
        self.log('rmse', self.rmse, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, movie_ids, _ = batch
        rating_pred = self(user_ids, movie_ids).squeeze()

        # Clamp the predicted ratings between 1.0 and 5.0
        rating_pred_clamped = self.clamp_ratings(rating_pred)

        return rating_pred_clamped

    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        
        return optimizer
