
import torch
from torch import nn

class Detector(torch.nn.Module):
    def __init__(self, in_dim, hidden=512, p=0.2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(p),
            torch.nn.Linear(hidden, hidden//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p),
            torch.nn.Linear(hidden//2, 2)
        )
    def forward(self, x):
        return self.net(x)