import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule


class CollaborativeFilteringModel(LightningModule):
    """Matrix factorization for dyadic regression.

    Predictions are ``dot(user_emb, item_emb) + user_bias + item_bias + global_bias``. Note that items are
    looked up in the user embedding and bias tables, so both tables are sized for the larger of the two id
    ranges. Training minimises the MSE of the unclamped prediction; validation and prediction clamp it to
    ``rating_range``.

    Args:
        num_users: Number of users.
        num_items: Number of items.
        embedding_dim: Embedding size.
        lr: AdamW learning rate.
        l2_reg: AdamW weight decay.
        rating_range: ``(min, max)`` observed value, used to clamp predictions.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 100,
        lr: float = 5e-4,
        l2_reg: float = 1e-5,
        rating_range: tuple = (1.0, 5.0),
    ):
        super().__init__()
        self.save_hyperparameters()

        num_embeddings = max(num_users, num_items)
        self.user_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.user_bias = nn.Embedding(num_embeddings, 1)
        self.global_bias = nn.Parameter(torch.tensor(0).float())

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)

        self.lr = lr
        self.l2_reg = l2_reg
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.min_rating, self.max_rating = rating_range

    def _clamp(self, x):
        return torch.clamp(x, min=self.min_rating, max=self.max_rating)

    def forward(self, user_ids, item_ids):
        dot_product = torch.sum(self.user_embedding(user_ids) * self.user_embedding(item_ids), dim=1, keepdim=True)
        return dot_product + self.user_bias(user_ids) + self.user_bias(item_ids) + self.global_bias

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings = batch
        loss = nn.functional.mse_loss(self(user_ids, item_ids).squeeze(), ratings)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings = batch
        rating_pred = self(user_ids, item_ids).squeeze()
        self.log("val_loss", nn.functional.mse_loss(rating_pred, ratings), on_step=False, on_epoch=True, prog_bar=True)
        self.rmse.update(self._clamp(rating_pred), ratings)
        self.log("rmse", self.rmse, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, _ = batch
        return self._clamp(self(user_ids, item_ids).squeeze())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
