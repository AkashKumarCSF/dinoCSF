import torch.nn as nn


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