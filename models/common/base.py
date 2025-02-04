import lightning as L
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Dict, Any

class BaseLightningModel(L.LightningModule):
    """Alap Lightning modell az összes modell számára"""
    def __init__(self, 
                 input_size: int,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.randn(1, 20, input_size)  # Autologging-hoz

    def configure_optimizers(self) -> Dict[str, Any]:
        opt = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(opt, patience=3),
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
