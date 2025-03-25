import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple CNN
class CNN(nn.Module):
    """Simple CNN model for FEMNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the neural network.

        Parameters
        ----------
            x (torch.Tensor): The input tensor.

        Returns
        -------
            torch.Tensor: The output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Alias for backward compatibility with your existing code
Net = CNN

class BiggerCNN(nn.Module):
    """A slightly deeper CNN for FEMNIST (28x28 grayscale)."""

    def __init__(self):
        super().__init__()
        # Convolution block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)  # Optional, can remove if BN complicates FL

        # Convolution block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        # Convolution block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After two pool operations on 28x28 input, 
        # the spatial dimension is roughly (28 // 2 // 2) = 7 on each side => 7x7.
        # Channels are 64, so flattened = 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 62)  # 62 classes in FEMNIST

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))  # or remove bn1(...) if not using BN
        x = self.pool(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        # If you want another pooling, you can do x = self.pool(x) here. 
        # But that might become too small, so we can skip it.

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
