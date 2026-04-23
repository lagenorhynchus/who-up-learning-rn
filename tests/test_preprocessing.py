"""Preprocessing verification suite. Run from project root: python -m pytest tests/"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocessing import (
    create_dual_stream_loaders,
    create_single_stream_loaders,
    check_dataloader,
    visualize_transformations,
    show_batch_grid,
    get_10_class_names,
)


def test_dataloader_shapes():
    train_loader, _, num_classes = create_dual_stream_loaders(
        batch_size=4, num_workers=0, use_10_class_subset=True
    )
    m_batch, p_batch, labels = next(iter(train_loader))
    assert m_batch.shape[1] == 1,   f"M-stream: expected 1 channel, got {m_batch.shape[1]}"
    assert p_batch.shape[1] == 3,   f"P-stream: expected 3 channels, got {p_batch.shape[1]}"
    assert m_batch.shape[2] == 128, f"Expected 128×128, got {m_batch.shape[2]}"
    assert num_classes == 10,       f"Expected 10 classes, got {num_classes}"
    print(f"M-stream shape: {m_batch.shape}")
    print(f"P-stream shape: {p_batch.shape}")
    print(f"Labels shape  : {labels.shape}")


def test_single_stream_loader():
    train_loader, _, num_classes = create_single_stream_loaders(
        batch_size=4, num_workers=0, use_10_class_subset=True
    )
    img_batch, labels = next(iter(train_loader))
    assert img_batch.shape[1] == 3, "Single-stream should have 3 channels"
    print(f"Single-stream batch shape: {img_batch.shape}")


def test_class_names():
    names = get_10_class_names()
    assert len(names) == 10
    print(f"10 class names: {names}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Preprocessing Verification Suite")
    print("=" * 60)

    print("\n[Test 1] Dataloader shapes")
    test_dataloader_shapes()

    print("\n[Test 2] Single-stream baseline loader")
    test_single_stream_loader()

    print("\n[Test 3] 10-class names")
    test_class_names()

    print("\n[Test 4] Generating visualizations")
    visualize_transformations(num_samples=3, save_path='output_preview.png')
    show_batch_grid(save_path='output_batch.png')

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
