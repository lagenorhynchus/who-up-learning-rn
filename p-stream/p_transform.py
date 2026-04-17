import torch
import torch.nn as nn
from torchvision import transforms


class PStreamTransform(nn.Module):
    """High-pass RGB filter for the parvocellular stream: x - GaussianBlur(x)."""

    def __init__(self, kernel_size=5, sigma=2.0):
        """Store a GaussianBlur op configured with the given kernel size and sigma."""
        super(PStreamTransform, self).__init__()
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, x):
        """Return x - blur(x). Accepts (3, H, W) or (B, 3, H, W). No clamping."""
        return x - self.blur(x)

    def visualize(self, x):
        """Return forward(x) centered at 0.5 so smooth regions display as mid-gray."""
        return (self.forward(x) * 0.5 + 0.5).clamp(0, 1)
