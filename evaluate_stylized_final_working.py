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
from sklearn.metrics import accuracy_score

from train import DualStreamCNN
from preprocessing.transforms import get_m_stream_transform, get_p_stream_transform

class StylizedDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
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
        return img, label


def extract_features(model, loader, device, m_transform, p_transform):
    """Extract features from all images"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels_batch) in enumerate(loader):
            # Apply M and P transforms to each image
            m_batch_list = []
            p_batch_list = []
            
            for img in images:
                # img is PIL Image, apply transforms directly
                m_img = m_transform(img)  # Returns tensor
                p_img = p_transform(img)  # Returns tensor
                m_batch_list.append(m_img)
                p_batch_list.append(p_img)
            
            m_batch = torch.stack(m_batch_list).to(device)
            p_batch = torch.stack(p_batch_list).to(device)
            labels_batch = labels_batch.to(device)
            
            # Forward through model
            m_out = model.m_stream(m_batch)
            p_out = model.p_stream(p_batch)
            combined = torch.cat([m_out, p_out], dim=1)
            
            features.append(combined.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    features = np.vstack(features)
    return features, np.array(labels)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = "/oscar/scratch/ojoseph3/data/stylized/dnn/session-1"
    
    # Create dataset (no transform here, we'll apply M/P transforms manually)
    dataset = StylizedDataset(data_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)  # num_workers=0 to avoid multiprocessing issues
    
    num_classes = len(dataset.class_to_idx)
    
    # Get M/P transforms
    m_transform = get_m_stream_transform(sigma=3.0, img_size=128)
    p_transform = get_p_stream_transform(sigma=3.0, strength=1.0, img_size=128)
    
    # Models to evaluate
    models_to_eval = [
        ("./experiments/exp1_default/checkpoint_epoch40.pt", "Exp1: Default (α=0.5 learnable)"),
        ("./experiments/exp2_shape_biased/checkpoint_epoch40.pt", "Exp2: Shape-Biased (α=0.8 fixed)"),
        ("./experiments/exp3_shape_biased_learnable/checkpoint_epoch40.pt", "Exp3: Shape-Biased Learnable (α=0.8 start)"),
        ("./experiments/exp4_extreme_shape/checkpoint_epoch40.pt", "Exp4: Extreme Shape (α=0.9 fixed)"),
    ]
    
    print("\n" + "="*70)
    print("STYLIZED-IMAGENET EVALUATION (Nearest Centroid on Extracted Features)")
    print("="*70)
    
    results = []
    for checkpoint_path, model_name in models_to_eval:
        if not os.path.exists(checkpoint_path):
            print(f"\nWarning: {checkpoint_path} not found, skipping...")
            continue
        
        print(f"\n{model_name}")
        print("-" * 50)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = DualStreamCNN(num_classes=100).to(device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("  Extracting features...")
        features, labels = extract_features(model, loader, device, m_transform, p_transform)
        print(f"  Features shape: {features.shape}")
        
        # Compute class means (nearest centroid)
        class_means = {}
        for class_idx in range(num_classes):
            class_features = features[labels == class_idx]
            if len(class_features) > 0:
                class_means[class_idx] = class_features.mean(axis=0)
        
        # Classify using nearest centroid
        predictions = []
        for feat in features:
            distances = []
            for class_idx, mean_vec in class_means.items():
                dist = np.linalg.norm(feat - mean_vec)
                distances.append((class_idx, dist))
            pred_class = min(distances, key=lambda x: x[1])[0]
            predictions.append(pred_class)
        
        accuracy = accuracy_score(labels, predictions) * 100
        results.append((model_name, accuracy))
        print(f"  Nearest Centroid Accuracy: {accuracy:.2f}%")
        
        # Also try simple baseline: random chance
        random_acc = (1.0 / num_classes) * 100
        print(f"  Random chance: {random_acc:.2f}%")
    
    print("\n" + "="*70)
    print("SUMMARY - STYLIZED-IMAGENET ACCURACY (Nearest Centroid)")
    print("="*70)
    for model_name, acc in results:
        print(f"{model_name:45s} {acc:.2f}%")
    
    # Determine best model
    best_model, best_acc = max(results, key=lambda x: x[1])
    print(f"\nBest performing model: {best_model} ({best_acc:.2f}%)")
    
    # Save results
    with open("stylized_results_final.txt", "w") as f:
        f.write("Stylized-ImageNet Evaluation Results (Nearest Centroid Classifier)\n")
        f.write("="*60 + "\n")
        f.write(f"Number of images: 800\n")
        f.write(f"Number of classes: {num_classes}\n\n")
        for model_name, acc in results:
            f.write(f"{model_name}: {acc:.2f}%\n")
        f.write(f"\nBest model: {best_model} ({best_acc:.2f}%)\n")

if __name__ == "__main__":
    main()
