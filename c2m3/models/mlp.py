import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple MLP
class MLP(nn.Module):
    """Simple MLP model for FEMNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters
        ----------
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Output tensor.
        """
        x = self.flatten(x)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x