import torch
import torch.nn as nn
from common.base import BaseLightningModel

class ICMLightning(BaseLightningModel):
    """Lightning ICM Modell residual kapcsolatokkal"""
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 **kwargs):
        super().__init__(input_size, **kwargs)
        
        self.block1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )
        
        self.residual = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        residual = self.residual(x)
        out = self.block1(x)
        out = self.block2(out)
        return self.fc_out(out + residual)