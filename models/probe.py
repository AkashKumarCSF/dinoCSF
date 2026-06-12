import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):

    def __init__(
        self,
        in_dim,
        num_classes
    ):
        super().__init__()

        self.fc = nn.Linear(
            in_dim,
            num_classes
        )

    def forward(self, x):

        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()

        # IMPORTANT for DINO / ViT features
        self.norm = nn.LayerNorm(in_dim)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x