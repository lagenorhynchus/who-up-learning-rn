import sys
sys.path.insert(0, '/oscar/scratch/ojoseph3/who-up-learning-rn')

import torch
import numpy as np
from sklearn.metrics import classification_report
from collections import defaultdict

from train import DualStreamCNN
from preprocessing.dataset import create_dual_stream_loaders

# CIFAR-100 class names (first 20 for reference)
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 
    'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 
    'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 
    'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 
    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load best model (Exp3)
    checkpoint_path = "./experiments/exp3_shape_biased_learnable/checkpoint_epoch40.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = DualStreamCNN(num_classes=100).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load CIFAR-100 validation set
    _, val_loader, _ = create_dual_stream_loaders(
        batch_size=64,
        use_10_class_subset=False,
        num_workers=4
    )
    
    print("Evaluating on CIFAR-100 validation set...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for m_batch, p_batch, labels in val_loader:
            m_batch = m_batch.to(device)
            p_batch = p_batch.to(device)
            labels = labels.to(device)
            
            outputs = model(m_batch, p_batch)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Get per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Classes we care about for comparison
    classes_of_interest = {
        'bicycle': 8,      # Index 8 in CIFAR-100
        'chair': 19,       # Index 19 in CIFAR-100  
        'car': None,       # Not in CIFAR-100 (closest: 'pickup_truck' index 57, 'streetcar' index 86)
        'truck': 57,       # 'pickup_truck' is index 57
        'boat': 13,        # 'boat' is index 13? Actually 'boat' is not precise, 'bridge'? 
    }
    
    print("\n" + "="*70)
    print("CIFAR-100 PER-CLASS ACCURACY FOR OVERLAPPING CLASSES")
    print("="*70)
    
    for class_name, class_idx in classes_of_interest.items():
        if class_idx is not None:
            mask = all_labels == class_idx
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
                print(f"{class_name:15s} (idx {class_idx:3d}): {class_acc:.2f}% ({mask.sum():4d} images)")
            else:
                print(f"{class_name:15s} (idx {class_idx:3d}): No images found")
        else:
            print(f"{class_name:15s}: Class not in CIFAR-100")
    
    # Find actual car-related classes
    print("\n" + "-"*70)
    print("Vehicle classes in CIFAR-100:")
    vehicle_indices = [57, 86, 48]  # pickup_truck, streetcar, motorcycle (index 48)
    vehicle_names = ['pickup_truck', 'streetcar', 'motorcycle']
    for name, idx in zip(vehicle_names, vehicle_indices):
        mask = all_labels == idx
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            print(f"  {name:15s}: {acc:.2f}% ({mask.sum():4d} images)")

if __name__ == "__main__":
    main()
