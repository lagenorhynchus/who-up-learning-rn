import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
class MagnocellularTransform:
    """
    Simulates the magnocellular visual pathway: low spatial frequency,
    achromatic, high temporal sensitivity channel.
    """
    def __init__(self):
        self.pipeline = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.GaussianBlur(kernel_size=5, sigma=1.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),  # maps to [-1, 1]
        ])
    def __call__(self, img):
        return self.pipeline(img)