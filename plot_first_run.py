import matplotlib.pyplot as plt

# Data from your first run (epochs 1-40)
epochs = list(range(1, 41))
train_loss = [4.1694, 3.5778, 3.2868, 3.1177, 3.0078, 2.9207, 2.8461, 2.7882, 2.7347, 2.6907, 2.6455, 2.6081, 2.5753, 2.5433, 2.5159, 2.4891, 2.4604, 2.4407, 2.4157, 2.3921, 2.3673, 2.3563, 2.3392, 2.3159, 2.2996, 2.2867, 2.2731, 2.2531, 2.2377, 2.2241, 2.2151, 2.1974, 2.1826, 2.1697, 2.1540, 2.1481, 2.1369, 2.1261, 2.1107, 2.1018]
val_loss = [4.5651, 3.8365, 3.5429, 4.6683, 3.9493, 3.3678, 3.0572, 3.5995, 5.2032, 4.2379, 3.5480, 3.7008, 3.1930, 2.9096, 2.9761, 4.2999, 3.6529, 3.9986, 3.1947, 2.9630, 3.0303, 3.1080, 3.0634, 2.6839, 3.3685, 2.9996, 3.0638, 3.2247, 3.3006, 2.8105, 2.8217, 3.0757, 4.1303, 2.9480, 2.6830, 3.1964, 3.0768, 3.1304, 3.5292, 3.2577]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Default Model (Alpha=0.28, Learned) - Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('default_train_loss.png', dpi=150)
print("Saved default_train_loss.png")
