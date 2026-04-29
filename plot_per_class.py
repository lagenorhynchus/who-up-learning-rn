import matplotlib.pyplot as plt
import numpy as np

# Data from your analysis
classes = ['bicycle', 'clock', 'chair', 'boat', 'car', 'truck', 'elephant', 
           'bird', 'bottle', 'knife', 'cat', 'dog', 'oven', 'airplane', 
           'bear', 'keyboard']
accuracies = [50.0, 40.0, 20.0, 12.0, 10.0, 10.0, 6.0, 4.0, 4.0, 4.0, 
              2.0, 2.0, 2.0, 0.0, 0.0, 0.0]

# Sort by accuracy
sorted_indices = np.argsort(accuracies)
classes_sorted = [classes[i] for i in sorted_indices]
accuracies_sorted = [accuracies[i] for i in sorted_indices]

# Create horizontal bar plot
plt.figure(figsize=(10, 8))
colors = ['green' if acc >= 20 else 'orange' if acc >= 10 else 'red' for acc in accuracies_sorted]
bars = plt.barh(classes_sorted, accuracies_sorted, color=colors)

# Add value labels
for bar, acc in zip(bars, accuracies_sorted):
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
             f'{acc:.0f}%', va='center', fontsize=10)

plt.xlabel('Accuracy (%)', fontsize=12)
plt.title('Per-Class Accuracy on Stylized-ImageNet (Exp3)', fontsize=14)
plt.axvline(x=6.25, color='red', linestyle='--', alpha=0.7, label='Random Chance (6.25%)')
plt.legend()
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=150)
plt.show()
print("Saved per_class_accuracy.png")
