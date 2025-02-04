# models/architectures/mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from models.common.base import BaseLightningModel

class MLPLightning(BaseLightningModel):
    """Többrétegű Perceptron idősor-predikcióhoz"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        **kwargs
    ):
        """
        Args:
            input_size (int): Bemeneti jellemzők száma (időbélyegek * jellemzők)
            hidden_size (int): Rejtett rétegek mérete
            num_layers (int): Rejtett rétegek száma
            dropout_rate (float): Dropout valószínűsége
            learning_rate (float): Optimalizáló tanulási ráta
        """
        super().__init__(input_size, learning_rate)
        
        # Rétegek dinamikus létrehozása
        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            out_features = hidden_size if i < num_layers-1 else 1
            
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
        
        # Utolsó aktivációs réteg eltávolítása
        layers = layers[:-1]
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bemenet átalakítása: (batch, seq_len, features) -> (batch, seq_len*features)
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
            
        return self.layers(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-5
        )