import sys
sys.path.insert(0, '/oscar/scratch/ojoseph3/who-up-learning-rn')

import torch
from PIL import Image
import os
import re
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from train import DualStreamCNN
from preprocessing.transforms import get_m_stream_transform, get_p_stream_transform

# Load dataset (reusing your existing code)
def load_dataset_with_classes(root_dir):
    image_paths = []
    labels = []
    class_names_list = []
    
    for fname in os.listdir(root_dir):
        if fname.endswith('.png'):
            match = re.search(r'_([a-z]+)_\d+_[a-z]+-', fname)
            if match:
                class_name = match.group(1)
                class_names_list.append(class_name)
                image_paths.append(os.path.join(root_dir, fname))
    
    # Create consistent class mapping
    unique_classes = sorted(set(class_names_list))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    labels = [class_to_idx[c] for c in class_names_list]
    
    return image_paths, labels, class_to_idx, idx_to_class

def extract_features(model, image_paths, device, m_transform, p_transform):
    model.eval()
    features = []
    
    for img_path in tqdm(image_paths, desc="  Extracting"):
        img = Image.open(img_path).convert('RGB')
        m_img = m_transform(img).unsqueeze(0).to(device)
        p_img = p_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            m_out = model.m_backbone(m_img)
            p_out = model.p_backbone(p_img)
            combined = torch.cat([m_out, p_out], dim=1)
        
        features.append(combined.cpu().numpy().flatten())
    
    return np.vstack(features)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "/oscar/scratch/ojoseph3/data/stylized/dnn/session-1"
    
    image_paths, labels, class_to_idx, idx_to_class = load_dataset_with_classes(data_path)
    num_classes = len(class_to_idx)
    
    print(f"Classes: {list(class_to_idx.keys())}")
    print(f"Images per class:")
    for class_name, idx in class_to_idx.items():
        count = labels.count(idx)
        print(f"  {class_name}: {count} images")
    
    m_transform = get_m_stream_transform(sigma=3.0, img_size=128)
    p_transform = get_p_stream_transform(sigma=3.0, strength=1.0, img_size=128)
    
    # Use best model (Exp3)
    checkpoint_path = "./experiments/exp3_shape_biased_learnable/checkpoint_epoch40.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DualStreamCNN(num_classes=100).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nExtracting features from best model (Exp3)...")
    features = extract_features(model, image_paths, device, m_transform, p_transform)
    
    # Compute class centroids
    class_means = {}
    for class_idx in range(num_classes):
        class_features = features[np.array(labels) == class_idx]
        if len(class_features) > 0:
            class_means[class_idx] = class_features.mean(axis=0)
    
    # Classify and compute per-class accuracy
    predictions = []
    for feat in features:
        distances = [(c, np.linalg.norm(feat - mean)) for c, mean in class_means.items()]
        pred_class = min(distances, key=lambda x: x[1])[0]
        predictions.append(pred_class)
    
    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    class_accuracies = {}
    for class_idx in range(num_classes):
        mask = np.array(labels) == class_idx
        if mask.sum() > 0:
            class_acc = np.mean(np.array(predictions)[mask] == class_idx) * 100
            class_name = idx_to_class[class_idx]
            class_accuracies[class_name] = class_acc
            print(f"  {class_name:15s}: {class_acc:.2f}% ({mask.sum():3d} images)")
    
    # Best and worst classes
    sorted_acc = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    print("\nBest performing classes:")
    for class_name, acc in sorted_acc[:5]:
        print(f"  {class_name}: {acc:.2f}%")
    
    print("\nWorst performing classes:")
    for class_name, acc in sorted_acc[-5:]:
        print(f"  {class_name}: {acc:.2f}%")

if __name__ == "__main__":
    main()
