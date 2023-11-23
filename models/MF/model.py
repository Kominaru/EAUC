from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics


class CollaborativeFilteringModel(LightningModule):
    """
    Collaborative filtering model that predicts the rating of a item for a user
    Ratings are computed as:
        r_hat = dot(user_embedding, item_embedding) + user_bias + item_bias + global_bias

    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 100,
        lr: float = 5e-4,
        l2_reg: float = 1e-5,
        dropout: float = 0.0,
        rating_range: tuple = (1.0, 5.0),
    ):
        """
        Initializes a Collaborative Filtering model for item ratings prediction

        Args:
            num_users (int): Number of users in the dataset
            num_items (int): Number of items in the dataset
            embedding_dim (int): Embedding size for user and item
            lr (float): Learning rate
            l2_reg (float): L2 regularization coefficient

        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0).float())

        self.user_dropout = nn.Dropout(dropout)
        self.item_dropout = nn.Dropout(dropout)

        self.lr = lr
        self.l2_reg = l2_reg

        self.loss_fn = nn.MSELoss()

        # Initialize embedding weights with Xavier distribution
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Initialize bias weights to zero
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        self.rmse = torchmetrics.MeanSquaredError(squared=False)

        self.clamp_ratings = lambda x: torch.clamp(x, min=rating_range[0], max=rating_range[1])

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        user_embeds = self.user_dropout(user_embeds)
        item_embeds = self.item_dropout(item_embeds)

        dot_product = torch.sum(user_embeds * item_embeds, dim=1, keepdim=True)

        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)

        prediction = (
            dot_product + user_bias + item_bias + self.global_bias
        )  # We add the user and item biases and the global bias

        return prediction

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings = batch
        rating_pred = self(user_ids, item_ids).squeeze()

        # Compute Mean Squared Error (MSE) loss
        # Notice we do not clamp the ratings
        loss = nn.MSELoss()(rating_pred, ratings)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings = batch
        rating_pred = self(user_ids, item_ids).squeeze()

        loss = nn.MSELoss()(rating_pred, ratings)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Clamp the predicted ratings for RMSE computation
        rating_pred_clamped = self.clamp_ratings(rating_pred)

        self.rmse.update(rating_pred_clamped, ratings)
        self.log("rmse", self.rmse, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, _ = batch
        rating_pred = self(user_ids, item_ids).squeeze()

        # Clamp the predicted ratings between 1.0 and 5.0
        rating_pred_clamped = self.clamp_ratings(rating_pred)

        return rating_pred_clamped

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)

        return optimizer
