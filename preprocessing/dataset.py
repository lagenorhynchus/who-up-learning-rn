"""
Dual-stream dataset loader that returns both M-stream and P-stream versions
of each image. Supports 10-class subset of CIFAR-100 for initial development.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CIFAR100
from .transforms import (
    get_m_stream_transform, 
    get_p_stream_transform, 
    get_standard_transform
)


class DualStreamDataset(Dataset):
    """
    Returns (m_image, p_image, label) for each sample.
    M-stream: low-pass grayscale, P-stream: high-pass color.
    """
    def __init__(self, base_dataset, m_transform, p_transform):
        self.base_dataset = base_dataset
        self.m_transform = m_transform
        self.p_transform = p_transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        m_img = self.m_transform(img)
        p_img = self.p_transform(img)
        return m_img, p_img, label


class SingleStreamDataset(Dataset):
    """Standard dataset for baseline comparison. Returns (image, label)."""
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = self.transform(img)
        return img, label


def get_10_class_subset(dataset, classes_to_keep=None):
    """
    Extract a 10-class subset from CIFAR-100.
    Default: first 10 classes (0-9).
    """
    if classes_to_keep is None:
        classes_to_keep = list(range(10))
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes_to_keep]
    return Subset(dataset, indices)


def create_dual_stream_loaders(
    data_dir='./data',
    batch_size=64,
    num_workers=4,
    img_size=128,
    sigma=3.0,
    strength=1.0,
    use_10_class_subset=True
):
    """
    Create train and validation DataLoaders for dual-stream model.
    
    Args:
        use_10_class_subset: True for 10 classes (development), False for full 100 classes (final)
    
    Returns:
        train_loader, val_loader, num_classes
    """
    m_transform = get_m_stream_transform(sigma=sigma, img_size=img_size)
    p_transform = get_p_stream_transform(sigma=sigma, strength=strength, img_size=img_size)
    
    train_full = CIFAR100(root=data_dir, train=True, download=True)
    val_full = CIFAR100(root=data_dir, train=False, download=True)
    
    if use_10_class_subset:
        train_base = get_10_class_subset(train_full)
        val_base = get_10_class_subset(val_full)
        num_classes = 10
        print(f"10-class mode. Train: {len(train_base)}, Val: {len(val_base)}")
    else:
        train_base = train_full
        val_base = val_full
        num_classes = 100
        print(f"100-class mode. Train: {len(train_base)}, Val: {len(val_base)}")
    
    train_dataset = DualStreamDataset(train_base, m_transform, p_transform)
    val_dataset = DualStreamDataset(val_base, m_transform, p_transform)
    
    # Check if running on MPS (Apple Silicon) to disable pin_memory
    if torch.backends.mps.is_available():
        use_pin_memory = False
    else:
        use_pin_memory = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, num_classes


def create_single_stream_loaders(
    data_dir='./data',
    batch_size=64,
    num_workers=4,
    img_size=128,
    use_10_class_subset=True
):
    """Create DataLoaders for standard single-stream baseline."""
    transform = get_standard_transform(img_size=img_size)
    
    train_full = CIFAR100(root=data_dir, train=True, download=True)
    val_full = CIFAR100(root=data_dir, train=False, download=True)
    
    if use_10_class_subset:
        train_base = get_10_class_subset(train_full)
        val_base = get_10_class_subset(val_full)
        num_classes = 10
    else:
        train_base = train_full
        val_base = val_full
        num_classes = 100
    
    train_dataset = SingleStreamDataset(train_base, transform)
    val_dataset = SingleStreamDataset(val_base, transform)
    
    # Check if running on MPS (Apple Silicon) to disable pin_memory
    if torch.backends.mps.is_available():
        use_pin_memory = False
    else:
        use_pin_memory = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, num_classes


def get_10_class_names():
    """Returns names of first 10 CIFAR-100 classes for interpretability."""
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
        'bed', 'bee', 'beetle', 'bicycle', 'bowl'
    ]
