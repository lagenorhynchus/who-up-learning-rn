import torch
import torch.nn as nn


class PStreamBackbone(nn.Module):
    """
    Lightweight CNN mirroring the parvocellular (P) visual pathway.

    Input  : (B, 3, H, W)  — high-pass filtered RGB
    Output : (B, 128)       — flat feature vector, no classification head

    Architecture (input-size agnostic via AdaptiveAvgPool2d):

        Block 1  Conv 3→32,  MaxPool2d(2)
        Block 2  Conv 32→64, MaxPool2d(2)
        Block 3  Conv 64→128,AdaptiveAvgPool2d(1)
        Flatten                         →  (B, 128)

    Biological grounding:
        Chromatic (3-ch RGB), high spatial frequency (upstream high-pass filter),
        fine-detail sensitivity — no FC head, feeds fusion layer.
    """

    def __init__(self) -> None:
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor  shape (B, 3, H, W)

        Returns
        -------
        torch.Tensor  shape (B, 128)
        """
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

        return x3.reshape(x3.shape[0], -1)  # (B, 128)
