import torch
from src.ml.models.base import TorchBaseModel
from torch.nn import functional as F

class LogisticRegressor(TorchBaseModel):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegressor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        return x.squeeze()

class MLP(TorchBaseModel):
    def __init__(self, input_dim: int, n_hidden_layers: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        return x.squeeze()
