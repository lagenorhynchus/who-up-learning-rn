"""
Image transformation pipelines for M-stream (low-pass grayscale)
and P-stream (high-pass color).
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import torch
import torchvision.transforms as transforms

from models.p_transform import PStreamTransform


class LowPassGrayscale:
    """
    Low-pass filter + grayscale for M-stream (PIL → PIL).
    Simulates magnocellular pathway: global structure, contrast, no fine detail.
    """
    def __init__(self, sigma: float = 3.0) -> None:
        self.sigma = sigma

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)

        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            blurred = np.zeros_like(img_np)
            for c in range(3):
                blurred[:, :, c] = gaussian_filter(img_np[:, :, c], sigma=self.sigma)
        else:
            blurred = gaussian_filter(img_np, sigma=self.sigma)

        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        return Image.fromarray(blurred).convert('L')


# PStreamTransform replaces the old numpy HighPassColor:
# it operates on tensors (post-ToTensor) so it is GPU-compatible and differentiable,
# and it is identical to what PStreamBackbone expects at runtime.
_p_transform_module = PStreamTransform(kernel_size=5, sigma=2.0)


def get_m_stream_transform(sigma: float = 3.0, img_size: int = 128):
    return transforms.Compose([
        LowPassGrayscale(sigma=sigma),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])


def get_p_stream_transform(sigma: float = 3.0, img_size: int = 128, **_):
    # sigma arg kept for API compatibility; PStreamTransform uses its own default
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        _p_transform_module,   # high-pass applied after normalisation
    ])


def get_standard_transform(img_size: int = 128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
