import sys
sys.path.insert(0, '/oscar/scratch/ojoseph3/who-up-learning-rn')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import re

# Import from your project
from train import DualStreamCNN
from preprocessing.transforms import get_m_stream_transform, get_p_stream_transform

class StylizedDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        # Extract unique classes from filenames
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
        
        print(f"Loaded {len(self.image_paths)} images from {len(self.class_to_idx)} classes")
        print(f"Classes: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        return img, label


def evaluate_model(model, loader, device, m_transform, p_transform, model_name):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            m_batch = torch.stack([m_transform(img) for img in images])
            p_batch = torch.stack([p_transform(img) for img in images])
            
            m_batch = m_batch.to(device)
            p_batch = p_batch.to(device)
            labels = labels.to(device)
            
            outputs = model(m_batch, p_batch)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {total} images...")
    
    return 100.0 * correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = "/oscar/scratch/ojoseph3/data/stylized/dnn/session-1"
    
    dataset = StylizedDataset(data_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    num_classes = len(dataset.class_to_idx)
    
    m_transform = get_m_stream_transform(sigma=3.0, img_size=128)
    p_transform = get_p_stream_transform(sigma=3.0, strength=1.0, img_size=128)
    
    models_to_eval = [
        ("./experiments/exp1_default/checkpoint_epoch40.pt", "Exp1: Default (α=0.5 learnable)"),
        ("./experiments/exp2_shape_biased/checkpoint_epoch40.pt", "Exp2: Shape-Biased (α=0.8 fixed)"),
        ("./experiments/exp3_shape_biased_learnable/checkpoint_epoch40.pt", "Exp3: Shape-Biased Learnable (α=0.8 start)"),
        ("./experiments/exp4_extreme_shape/checkpoint_epoch40.pt", "Exp4: Extreme Shape (α=0.9 fixed)"),
    ]
    
    print("\n" + "="*60)
    print(f"EVALUATING ON STYLIZED-IMAGENET ({len(dataset)} images, {num_classes} classes)")
    print("="*60)
    
    results = []
    for checkpoint_path, model_name in models_to_eval:
        if not os.path.exists(checkpoint_path):
            print(f"\nWarning: {checkpoint_path} not found, skipping...")
            continue
        
        print(f"\nLoading {model_name}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = DualStreamCNN(num_classes=num_classes).to(device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"Evaluating {model_name}...")
        acc = evaluate_model(model, loader, device, m_transform, p_transform, model_name)
        results.append((model_name, acc))
        print(f"  Accuracy: {acc:.2f}%")
    
    print("\n" + "="*60)
    print("SUMMARY - STYLIZED-IMAGENET ACCURACY")
    print("="*60)
    for model_name, acc in results:
        print(f"{model_name:45s} {acc:.2f}%")
    
    with open("stylized_results.txt", "w") as f:
        f.write("Stylized-ImageNet Evaluation Results\n")
        f.write("="*50 + "\n")
        for model_name, acc in results:
            f.write(f"{model_name}: {acc:.2f}%\n")
        f.write(f"\nTotal images: {len(dataset)}\n")
        f.write(f"Number of classes: {num_classes}\n")

if __name__ == "__main__":
    main()
