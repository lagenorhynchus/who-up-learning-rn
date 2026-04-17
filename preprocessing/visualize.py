"""
Visualization utilities to verify preprocessing is working correctly.
"""

import matplotlib.pyplot as plt
import numpy as np
from .transforms import LowPassGrayscale, HighPassColor
from .dataset import create_dual_stream_loaders, get_10_class_names


def visualize_transformations(num_samples=4, save_path='preview.png'):
    """Show original, M-stream, and P-stream versions of the same images."""
    from torchvision.datasets import CIFAR100
    
    full_dataset = CIFAR100(root='./data', train=True, download=True)
    indices = [i for i, (_, label) in enumerate(full_dataset) if label < 10]
    
    m_transform = LowPassGrayscale(sigma=3.0)
    p_transform = HighPassColor(sigma=3.0, strength=1.0)
    class_names = get_10_class_names()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3*num_samples))
    
    for i in range(num_samples):
        idx = indices[i]
        img, label = full_dataset[idx]
        class_name = class_names[label]
        
        m_img = m_transform(img)
        p_img = p_transform(img)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original: {class_name}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(m_img, cmap='gray')
        axes[i, 1].set_title(f'M-Stream: {class_name}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(p_img)
        axes[i, 2].set_title(f'P-Stream: {class_name}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved preview to {save_path}")


def check_dataloader():
    """Verify dataloader returns correct shapes."""
    train_loader, val_loader, num_classes = create_dual_stream_loaders(
        batch_size=4,
        num_workers=0,
        use_10_class_subset=True
    )
    
    m_batch, p_batch, labels = next(iter(train_loader))
    
    print("=" * 50)
    print("DATALOADER VERIFICATION")
    print("=" * 50)
    print(f"M-stream shape: {m_batch.shape}")  # [4, 1, 128, 128]
    print(f"P-stream shape: {p_batch.shape}")  # [4, 3, 128, 128]
    print(f"Labels shape: {labels.shape}")      # [4]
    print(f"Number of classes: {num_classes}")  # 10
    print("=" * 50)
    
    assert m_batch.shape[1] == 1, f"Expected 1 channel, got {m_batch.shape[1]}"
    assert p_batch.shape[1] == 3, f"Expected 3 channels, got {p_batch.shape[1]}"
    assert m_batch.shape[2] == 128, f"Expected 128x128, got {m_batch.shape[2]}"
    assert num_classes == 10, f"Expected 10 classes, got {num_classes}"
    
    print("All tests passed.")


def show_batch_grid(save_path='batch_preview.png'):
    """Display a grid of M-stream and P-stream batches side by side."""
    train_loader, _, _ = create_dual_stream_loaders(batch_size=8, num_workers=0)
    m_batch, p_batch, labels = next(iter(train_loader))
    
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    for i in range(8):
        m_img = m_batch[i].squeeze(0).numpy()
        axes[0, i].imshow(m_img, cmap='gray')
        axes[0, i].set_title(f'M-Stream\nLabel: {labels[i].item()}')
        axes[0, i].axis('off')
        
        p_img = p_batch[i].permute(1, 2, 0).numpy()
        p_img = p_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        p_img = np.clip(p_img, 0, 1)
        axes[1, i].imshow(p_img)
        axes[1, i].set_title(f'P-Stream\nLabel: {labels[i].item()}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved batch preview to {save_path}")


if __name__ == "__main__":
    check_dataloader()
    visualize_transformations()
    show_batch_grid()
