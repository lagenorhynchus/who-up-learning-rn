"""Pull one CIFAR-100 (10-class subset) batch through the dataloader and DualStreamCNN; report shapes + top-1 picks."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import DualStreamCNN
from preprocessing import create_dual_stream_loaders


def main():
    train_loader, _, num_classes = create_dual_stream_loaders(
        batch_size=16,
        use_10_class_subset=True,
        num_workers=0,
    )
    m_batch, p_batch, labels = next(iter(train_loader))

    model = DualStreamCNN(num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        logits = model(m_batch, p_batch)

    preds = logits.argmax(dim=1)
    print(f"m_batch: {tuple(m_batch.shape)}   p_batch: {tuple(p_batch.shape)}")
    print(f"logits:  {tuple(logits.shape)}    num_classes: {num_classes}")
    print(f"alpha:   {model.alpha.item():.3f}  (learnable={model.learnable_alpha})")
    print(f"labels:  {labels.tolist()}")
    print(f"preds:   {preds.tolist()}   (random init, so matches are coincidence)")


if __name__ == "__main__":
    main()
