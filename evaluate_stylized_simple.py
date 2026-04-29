import sys
sys.path.insert(0, '/oscar/scratch/ojoseph3/who-up-learning-rn')

import torch
from torchvision import transforms
from PIL import Image
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from train import DualStreamCNN
from preprocessing.transforms import get_m_stream_transform, get_p_stream_transform

def load_dataset(root_dir):
    """Load all image paths and labels without using DataLoader"""
    image_paths = []
    labels = []
    class_names = set()
    
    for fname in os.listdir(root_dir):
        if fname.endswith('.png'):
            match = re.search(r'_([a-z]+)_\d+_[a-z]+-', fname)
            if match:
                class_name = match.group(1)
                class_names.add(class_name)
    
    class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
    
    for fname in os.listdir(root_dir):
        if fname.endswith('.png'):
            match = re.search(r'_([a-z]+)_\d+_[a-z]+-', fname)
            if match:
                class_name = match.group(1)
                image_paths.append(os.path.join(root_dir, fname))
                labels.append(class_to_idx[class_name])
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"Loaded {len(image_paths)} images from {len(class_to_idx)} classes")
    print(f"Classes: {list(class_to_idx.keys())}")
    
    return image_paths, labels, class_to_idx, idx_to_class


def extract_features_single(model, image_path, device, m_transform, p_transform):
    """Extract features from a single image"""
    img = Image.open(image_path).convert('RGB')
    
    m_img = m_transform(img).unsqueeze(0).to(device)
    p_img = p_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        m_out = model.m_stream(m_img)
        p_out = model.p_stream(p_img)
        combined = torch.cat([m_out, p_out], dim=1)
    
    return combined.cpu().numpy().flatten()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = "/oscar/scratch/ojoseph3/data/stylized/dnn/session-1"
    
    # Load dataset
    image_paths, labels, class_to_idx, idx_to_class = load_dataset(data_path)
    num_classes = len(class_to_idx)
    
    # Get transforms
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
    print("STYLIZED-IMAGENET EVALUATION (Nearest Centroid Classifier)")
    print("="*70)
    
    results = []
    for checkpoint_path, model_name in models_to_eval:
        if not os.path.exists(checkpoint_path):
            print(f"\nWarning: {checkpoint_path} not found, skipping...")
            continue
        
        print(f"\n{model_name}")
        print("-" * 50)
        print("  Loading model...")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = DualStreamCNN(num_classes=100).to(device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        print("  Extracting features from 800 images...")
        features = []
        for img_path in tqdm(image_paths, desc="  Processing"):
            feat = extract_features_single(model, img_path, device, m_transform, p_transform)
            features.append(feat)
        
        features = np.vstack(features)
        labels_np = np.array(labels)
        
        print(f"  Features shape: {features.shape}")
        
        # Compute class means (nearest centroid)
        class_means = {}
        for class_idx in range(num_classes):
            class_features = features[labels_np == class_idx]
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
        
        accuracy = accuracy_score(labels_np, predictions) * 100
        results.append((model_name, accuracy))
        print(f"  Nearest Centroid Accuracy: {accuracy:.2f}%")
        
        random_acc = (1.0 / num_classes) * 100
        print(f"  Random chance: {random_acc:.2f}%")
    
    print("\n" + "="*70)
    print("SUMMARY - STYLIZED-IMAGENET ACCURACY")
    print("="*70)
    for model_name, acc in results:
        print(f"{model_name:45s} {acc:.2f}%")
    
    if results:
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
        if results:
            f.write(f"\nBest model: {best_model} ({best_acc:.2f}%)\n")
    
    print("\nResults saved to stylized_results_final.txt")

if __name__ == "__main__":
    main()
