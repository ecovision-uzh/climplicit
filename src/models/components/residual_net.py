import torch
from torch import nn


class ResidLayer(torch.nn.Module):
    """Residual block used in Residual_Net"""
    def __init__(self, hidden_dim, batchnorm):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim) if batchnorm else torch.nn.Identity(),
        )
        
    def forward(self, x):
        b = self.layers(x)
        x = x + b
        return x


class Residual_Net(torch.nn.Module):
    """Base Residual net from the SINR paper"""
    def __init__(self, input_len=11, hidden_dim = 64, layers = 2, out_dim=256, batchnorm=True):
        super().__init__()
        
        self.res_layers = torch.nn.Sequential(
            torch.nn.Linear(input_len, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim) if batchnorm else torch.nn.Identity(),
            *[ResidLayer(hidden_dim, batchnorm) for i in range(layers)]
        )
        
        self.final_transform = torch.nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.res_layers(x)
        x = self.final_transform(x)
        return x



if __name__ == "__main__":
    _ = Residual_Net()
