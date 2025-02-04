import torch
import torch.nn as nn
from common.base import BaseLightningModel
from typing import List

class NBEATSLightning(BaseLightningModel):
    """Lightning N-BEATS Modell az eredeti papír szerint"""
    def __init__(self,
                 input_size: int,
                 stack_types: List[str] = ["trend", "seasonality"],
                 num_blocks: int = 3,
                 num_layers: int = 4,
                 layer_size: int = 256,
                 **kwargs):
        super().__init__(input_size, **kwargs)
        
        self.stacks = nn.ModuleList([
            self.create_stack(stack_type, num_blocks, num_layers, layer_size)
            for stack_type in stack_types
        ])
        
        self.fc_out = nn.Linear(len(stack_types)*layer_size, 1)

    def create_stack(self, 
                    stack_type: str,
                    num_blocks: int,
                    num_layers: int,
                    layer_size: int) -> nn.ModuleList:
        blocks = []
        for _ in range(num_blocks):
            block = NBEATSBlock(stack_type, layer_size, num_layers, self.hparams.input_size)
            blocks.append(block)
        return nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        forecasts = []
        for stack in self.stacks:
            residual = x.clone()
            stack_forecast = 0
            for block in stack:
                backcast, forecast = block(residual)
                residual -= backcast
                stack_forecast += forecast
            forecasts.append(stack_forecast)
        return self.fc_out(torch.cat(forecasts, dim=1))
    
class NBEATSBlock(nn.Module):
    """N-BEATS blokk implementáció"""
    def __init__(self,
                 stack_type: str,
                 layer_size: int,
                 num_layers: int,
                 input_size: int):
        super().__init__()
        
        self.fc_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(layer_size, layer_size),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])
        
        if stack_type == "trend":
            self.theta_size = 2 * (3 + 1)  # 3. fokú polinom + időszelep
        else:
            self.theta_size = 2 * 4  # 4 harmonikus komponens
            
        self.fc_theta = nn.Linear(layer_size, self.theta_size)
        self.backcast_fc = nn.Linear(self.theta_size, input_size)
        self.forecast_fc = nn.Linear(self.theta_size, layer_size)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.fc_layers(x)
        theta = self.fc_theta(x)
        return self.backcast_fc(theta), self.forecast_fc(theta)    