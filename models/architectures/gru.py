import torch
import torch.nn as nn
from common.base import BaseLightningModel

class GRULightning(BaseLightningModel):
    """Lightning GRU Modell a korábbi architektúrával"""
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 **kwargs):
        super().__init__(input_size, **kwargs)
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size*2,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        return self.fc(attn_out[:, -1, :])