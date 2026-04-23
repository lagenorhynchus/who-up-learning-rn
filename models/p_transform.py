import torch
import torch.nn as nn
from torchvision import transforms


class PStreamTransform(nn.Module):
    """
    High-pass RGB filter for the parvocellular stream: output = x - GaussianBlur(x).

    Operates on tensors (post-ToTensor), making it GPU-compatible and differentiable,
    unlike the numpy/scipy approach in preprocessing/transforms.py which runs on PIL images.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 2.0) -> None:
        super().__init__()
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (3, H, W) or (B, 3, H, W)

        Returns
        -------
        torch.Tensor  same shape — high-frequency residual, not clamped
        """
        return x - self.blur(x)

    def visualize(self, x: torch.Tensor) -> torch.Tensor:
        """Return forward(x) centered at 0.5 so smooth regions display as mid-gray."""
        return (self.forward(x) * 0.5 + 0.5).clamp(0, 1)
