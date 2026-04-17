"""
Standalone test script for Person 3. Run from project root.
"""

import sys
sys.path.append('.')

from preprocessing import (
    create_dual_stream_loaders,
    create_single_stream_loaders,
    check_dataloader,
    visualize_transformations,
    show_batch_grid,
    get_10_class_names
)


def main():
    print("\n" + "="*60)
    print("Person 3: Preprocessing Verification Suite")
    print("="*60)
    
    print("\n[Test 1] Dataloader shapes")
    check_dataloader()
    
    print("\n[Test 2] 10-class names")
    names = get_10_class_names()
    for i, name in enumerate(names):
        print(f"  Class {i}: {name}")
    
    print("\n[Test 3] Generating visualizations")
    visualize_transformations(num_samples=3, save_path='output_preview.png')
    show_batch_grid(save_path='output_batch.png')
    
    print("\n[Test 4] Single-stream baseline loader")
    train_loader_single, val_loader_single, num_classes_single = create_single_stream_loaders(
        use_10_class_subset=True
    )
    img_batch, labels_batch = next(iter(train_loader_single))
    print(f"Single-stream batch shape: {img_batch.shape}")
    print(f"Single-stream labels shape: {labels_batch.shape}")
    assert img_batch.shape[1] == 3, "Single-stream should have 3 channels"
    print("Single-stream loader verified.")
    
    print("\n" + "="*60)
    print("All tests passed. Preprocessing ready for Person 1 and Person 2.")
    print("="*60)
    
    print("\n" + "="*60)
    print("Handoff Instructions for Person 1 and Person 2")
    print("="*60)
    print("""
Import the dataloader:

    from preprocessing import create_dual_stream_loaders
    
    train_loader, val_loader, num_classes = create_dual_stream_loaders(
        batch_size=64,
        use_10_class_subset=True   # False for 100 classes
    )
    
    for m_batch, p_batch, labels in train_loader:
        # m_batch.shape = [batch, 1, 128, 128]  -> M-stream input
        # p_batch.shape = [batch, 3, 128, 128]  -> P-stream input
        # labels.shape = [batch]

Architecture requirements:
- M-stream: 3 conv layers (32 -> 64 -> 128 channels) -> output 128-dim vector
- P-stream: 3 conv layers (32 -> 64 -> 128 channels) -> output 128-dim vector
- Concatenate: 128 + 128 = 256-dim vector
- Final linear: 256 -> num_classes (10 or 100)
""")


if __name__ == "__main__":
    main()
