import sys
sys.path.insert(0, '/oscar/scratch/ojoseph3/who-up-learning-rn')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score

from train import DualStreamCNN
from preprocessing.transforms import get_m_stream_transform, get_p_stream_transform

class StylizedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        class_names = set()
        for fname in os.listdir(root_dir):
            if fname.endswith('.png'):
                match = re.search(r'_([a-z]+)_\d+_[a-z]+-', fname)
                if match:
                    class_name = match.group(1)
                    class_names.add(class_name)
        
        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
        
        for fname in os.listdir(root_dir):
            if fname.endswith('.png'):
                match = re.search(r'_([a-z]+)_\d+_[a-z]+-', fname)
                if match:
                    class_name = match.group(1)
                    self.image_paths.append(os.path.join(root_dir, fname))
                    self.labels.append(self.class_to_idx[class_name])
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(f"Loaded {len(self.image_paths)} images from {len(self.class_to_idx)} classes")
        print(f"Classes: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        class_name = self.idx_to_class[label]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, class_name


def extract_features(model, loader, device, m_transform, p_transform):
    """Extract features (before the final classifier) from all images"""
    model.eval()
    features = []
    labels = []
    class_names = []
    
    with torch.no_grad():
        for batch_idx, (images, labels_batch, class_names_batch) in enumerate(loader):
            # Apply M and P transforms to each image in batch
            m_batch = torch.stack([m_transform(img) for img in images])
            p_batch = torch.stack([p_transform(img) for img in images])
            
            m_batch = m_batch.to(device)
            p_batch = p_batch.to(device)
            
            # Forward through both streams
            m_out = model.m_stream(m_batch)
            p_out = model.p_stream(p_batch)
            combined = torch.cat([m_out, p_out], dim=1)  # 256-dim feature
            
            features.append(combined.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())
            class_names.extend(class_names_batch)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    features = np.vstack(features)
    return features, np.array(labels), class_names


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = "/oscar/scratch/ojoseph3/data/stylized/dnn/session-1"
    
    # Transform to convert PIL to tensor (but not the M/P transforms yet)
    to_tensor = transforms.ToTensor()
    
    # Create dataset and loader
    dataset = StylizedDataset(data_path, transform=to_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    num_classes = len(dataset.class_to_idx)
    
    # Get M/P transforms (these will be applied manually in extract_features)
    m_transform = get_m_stream_transform(sigma=3.0, img_size=128)
    p_transform = get_p_stream_transform(sigma=3.0, strength=1.0, img_size=128)
    
    # Model to evaluate (use best model from Exp3)
    checkpoint_path = "./experiments/exp3_shape_biased_learnable/checkpoint_epoch40.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with its original number of classes (100)
    model = DualStreamCNN(num_classes=100).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully")
    
    # Extract features
    print("\nExtracting features from stylized images...")
    features, labels, class_names = extract_features(model, loader, device, m_transform, p_transform)
    
    print(f"\nExtracted features shape: {features.shape}")
    
    # For each class, compute the mean feature vector
    class_means = {}
    for class_idx in range(num_classes):
        class_features = features[labels == class_idx]
        if len(class_features) > 0:
            class_means[class_idx] = class_features.mean(axis=0)
    
    # Classify using nearest centroid
    predictions = []
    for i, feat in enumerate(features):
        distances = []
        for class_idx, mean_vec in class_means.items():
            dist = np.linalg.norm(feat - mean_vec)
            distances.append((class_idx, dist))
        pred_class = min(distances, key=lambda x: x[1])[0]
        predictions.append(pred_class)
    
    accuracy = accuracy_score(labels, predictions)
    print(f"\n{'='*60}")
    print(f"STYLIZED-IMAGENET EVALUATION (Nearest Centroid Classifier)")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_acc = accuracy_score(labels[mask], np.array(predictions)[mask])
            class_name = dataset.idx_to_class[class_idx]
            print(f"  {class_name:15s}: {class_acc * 100:.2f}% ({mask.sum():3d} images)")
    
    # Save results
    with open("stylized_results_centroid.txt", "w") as f:
        f.write("Stylized-ImageNet Evaluation Results (Nearest Centroid)\n")
        f.write("="*60 + "\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Per-class accuracy:\n")
        for class_idx in range(num_classes):
            mask = labels == class_idx
            if mask.sum() > 0:
                class_acc = accuracy_score(labels[mask], np.array(predictions)[mask])
                class_name = dataset.idx_to_class[class_idx]
                f.write(f"  {class_name}: {class_acc * 100:.2f}%\n")
    
    print("\nResults saved to stylized_results_centroid.txt")

if __name__ == "__main__":
    main()
