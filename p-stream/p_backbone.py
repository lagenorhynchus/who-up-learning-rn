import torch
import torch.nn as nn


class PStreamBackbone(nn.Module):
    """P-stream backbone: 3 conv blocks mapping high-pass RGB to a 128-dim feature vector."""

    def __init__(self):
        """Build conv / batch norm / relu / pool layers for three blocks 3 -> 32 -> 64 -> 128."""
        super(PStreamBackbone, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        """Run the three conv blocks on (B, 3, H, W) inputs and return a (B, 128) feature vector."""
        x1 = self.conv1(inputs)
        x1 = self.batch_norm1(x1)
        x1 = self.relu1(x1)
        x1 = self.pool1(x1)

        x2 = self.conv2(x1)
        x2 = self.batch_norm2(x2)
        x2 = self.relu2(x2)
        x2 = self.pool2(x2)

        x3 = self.conv3(x2)
        x3 = self.batch_norm3(x3)
        x3 = self.relu3(x3)
        x3 = self.pool3(x3)

        features = x3.reshape(x3.shape[0], -1)
        return features
