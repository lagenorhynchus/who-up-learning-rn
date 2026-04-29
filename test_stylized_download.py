from modelvshuman import datasets

print("Loading stylized dataset (first time - will download)...")
dataset = datasets.stylized(batch_size=16, num_workers=4)
print("Success! Dataset loaded.")
print(f"Dataset has {len(dataset.dataset)} images")

# Test one batch
for images, labels in dataset:
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    break
