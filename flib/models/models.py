import torch
import torch_geometric.nn
from flib.models import TorchBaseModel
from torch.nn import functional as F
from typing import Dict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

class GCN(TorchBaseModel):
    def __init__(self, input_dim:int, n_conv_layers:int, hidden_dim:int, output_dim:int, dropout:float=0.8):
        super().__init__()
        self.dropout = dropout
        self.input_layer = torch_geometric.nn.Linear(input_dim, hidden_dim)
        self.conv_layers = torch.nn.ModuleList([torch_geometric.nn.GCNConv(hidden_dim, hidden_dim) for _ in range(n_conv_layers)])
        self.layer_norms = torch.nn.ModuleList([torch_geometric.nn.LayerNorm(hidden_dim) for _ in range(n_conv_layers)])
        self.output_layer = torch_geometric.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x = self.input_layer(data.x)
        x = F.relu(x)
        for conv_layer, layer_norm in zip(self.conv_layers, self.layer_norms):
            x = conv_layer(x, data.edge_index)
            x = layer_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.output_layer(x)
        return x.squeeze()
    
    # Overriding method
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.grad.clone().detach() 
            for name, param in self.named_parameters() if 'layer_norms' not in name
        }

    # Overriding method
    def set_gradients(self, gradients: Dict[str, torch.Tensor], strict: bool = False):
        for name, param in self.named_parameters():
            if name in gradients and 'layer_norms' not in name:
                param.grad = gradients[name].clone().detach()
            elif strict:
                raise KeyError(f"Gradient for parameter '{name}' not found in provided dictionary.")
    
    # Overriding method
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.data.clone().detach() 
            for name, param in self.named_parameters() if 'layer_norms' not in name
        }

    # Overriding method
    def set_parameters(self, parameters: Dict[str, torch.Tensor], strict: bool = False):
        for name, param in self.named_parameters():
            if name in parameters and 'layer_norms' not in name:
                param.data = parameters[name].clone().detach()
            elif strict:
                raise KeyError(f"Parameter '{name}' not found in provided dictionary.")

class GAT(TorchBaseModel):
    def __init__(self, input_dim:int, n_conv_layers:int, hidden_dim:int, output_dim:int, dropout:float=0.2):
        super().__init__()
        self.dropout = dropout
        self.input_layer = torch_geometric.nn.Linear(input_dim, hidden_dim)
        self.conv_layers = torch.nn.ModuleList([torch_geometric.nn.GATConv(hidden_dim, hidden_dim) for _ in range(n_conv_layers)])
        self.layer_norms = torch.nn.ModuleList([torch_geometric.nn.LayerNorm(hidden_dim) for _ in range(n_conv_layers)])
        self.output_layer = torch_geometric.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x = self.input_layer(data.x)
        x = F.relu(x)
        for conv_layer, layer_norm in zip(self.conv_layers, self.layer_norms):
            x = conv_layer(x, data.edge_index)
            x = layer_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.output_layer(x)
        return x.squeeze()
    
    # Overriding method
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.grad.clone().detach() 
            for name, param in self.named_parameters() if 'layer_norms' not in name
        }

    # Overriding method
    def set_gradients(self, gradients: Dict[str, torch.Tensor], strict: bool = False):
        for name, param in self.named_parameters():
            if name in gradients and 'layer_norms' not in name:
                param.grad = gradients[name].clone().detach()
            elif strict:
                raise KeyError(f"Gradient for parameter '{name}' not found in provided dictionary.")
    
    # Overriding method
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.data.clone().detach() 
            for name, param in self.named_parameters() if 'layer_norms' not in name
        }

    # Overriding method
    def set_parameters(self, parameters: Dict[str, torch.Tensor], strict: bool = False):
        for name, param in self.named_parameters():
            if name in parameters and 'layer_norms' not in name:
                param.data = parameters[name].clone().detach()
            elif strict:
                raise KeyError(f"Parameter '{name}' not found in provided dictionary.")

class GraphSAGE(TorchBaseModel):
    def __init__(self, input_dim:int, n_conv_layers:int, hidden_dim:int, output_dim:int, dropout:float=0.2):
        super().__init__()
        self.dropout = dropout
        self.input_layer = torch_geometric.nn.Linear(input_dim, hidden_dim)
        self.conv_layers = torch.nn.ModuleList([torch_geometric.nn.SAGEConv(hidden_dim, hidden_dim) for _ in range(n_conv_layers)])
        self.layer_norms = torch.nn.ModuleList([torch_geometric.nn.LayerNorm(hidden_dim) for _ in range(n_conv_layers)])
        self.output_layer = torch_geometric.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x = self.input_layer(data.x)
        x = F.relu(x)
        for conv_layer, layer_norm in zip(self.conv_layers, self.layer_norms):
            x = conv_layer(x, data.edge_index)
            x = layer_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.output_layer(x)
        return x.squeeze()
    
    # Overriding method
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.grad.clone().detach() 
            for name, param in self.named_parameters() if 'layer_norms' not in name
        }

    # Overriding method
    def set_gradients(self, gradients: Dict[str, torch.Tensor], strict: bool = False):
        for name, param in self.named_parameters():
            if name in gradients and 'layer_norms' not in name:
                param.grad = gradients[name].clone().detach()
            elif strict:
                raise KeyError(f"Gradient for parameter '{name}' not found in provided dictionary.")
    
    # Overriding method
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.data.clone().detach() 
            for name, param in self.named_parameters() if 'layer_norms' not in name
        }

    # Overriding method
    def set_parameters(self, parameters: Dict[str, torch.Tensor], strict: bool = False):
        for name, param in self.named_parameters():
            if name in parameters and 'layer_norms' not in name:
                param.data = parameters[name].clone().detach()
            elif strict:
                raise KeyError(f"Parameter '{name}' not found in provided dictionary.")
