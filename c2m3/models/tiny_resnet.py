import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """A simplified ResNet Basic Block (no downsampling if not needed)."""
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels),
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = self.relu(out)
        return out


class TinyResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=62, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)  # Grayscale
        self.bn1 = norm_layer(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, 2, stride=1, norm_layer=norm_layer)  # 28x28
        self.layer2 = self._make_layer(block, 32, 2, stride=2, norm_layer=norm_layer)  # 14x14
        self.layer3 = self._make_layer(block, 64, 2, stride=2, norm_layer=norm_layer)  # 7x7

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride, norm_layer):
        layers = [block(self.in_channels, out_channels, stride, norm_layer)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, 1, norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
