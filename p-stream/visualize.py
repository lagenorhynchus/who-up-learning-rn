import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from p_transform import PStreamTransform


def main():
    """Render 4 CIFAR-100 test images next to their P-stream (high-pass) counterparts."""
    dataset = datasets.CIFAR100(root='../data', train=False, download=True)
    to_tensor = transforms.ToTensor()
    p_transform = PStreamTransform()
    indices = [0, 100, 200, 300]

    fig, axes = plt.subplots(len(indices), 2, figsize=(4, 8))
    for row, idx in enumerate(indices):
        img_pil, _ = dataset[idx]
        img = to_tensor(img_pil)
        filtered = p_transform.visualize(img)

        axes[row][0].imshow(img.permute(1, 2, 0).numpy())
        axes[row][1].imshow(filtered.permute(1, 2, 0).numpy())
        if row == 0:
            axes[row][0].set_title("Original")
            axes[row][1].set_title("P-Stream (high-pass)")
        axes[row][0].axis('off')
        axes[row][1].axis('off')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/p_stream_viz.png', dpi=150)
    print("Saved to outputs/p_stream_viz.png")


if __name__ == "__main__":
    main()
