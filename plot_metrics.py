import json
import matplotlib.pyplot as plt
import os

with open('models/skeleton_veloci/tcn_veloci_results.json', 'r') as f:
    data = json.load(f)

history = data['TCN_Veloci']['history']
epochs = range(1, len(history['train_loss']) + 1)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/skeleton_veloci/learning_curve.png', dpi=300)
print("Saved chart to models/skeleton_veloci/learning_curve.png")
