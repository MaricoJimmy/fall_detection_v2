"""
Evaluate LSTM tren test set (TCN da evaluate roi)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

CONFIG = {
    'data_dir': 'datasets/skeleton_windows',
    'output_dir': 'models/skeleton',
    'batch_size': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class FallLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])

def main():
    device = torch.device(CONFIG['device'])
    
    X_test = torch.FloatTensor(np.load(os.path.join(CONFIG['data_dir'], 'X_test.npy')))
    y_test = torch.FloatTensor(np.load(os.path.join(CONFIG['data_dir'], 'y_test.npy')))
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = FallLSTM(input_size=51, hidden_size=128, num_layers=2, dropout=0.3)
    checkpoint = torch.load(os.path.join(CONFIG['output_dir'], 'LSTM_best.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(predicted.flatten())
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    mr = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    print("="*70)
    print("  LSTM EVALUATION ON TEST SET")
    print("="*70)
    print(f"Confusion Matrix:")
    print(f"  TN: {tn:6d}  FP: {fp:6d}")
    print(f"  FN: {fn:6d}  TP: {tp:6d}")
    print()
    print(f"  Accuracy:          {accuracy*100:.2f}%")
    print(f"  Precision:         {precision*100:.2f}%")
    print(f"  Recall:            {recall*100:.2f}%")
    print(f"  Specificity:       {specificity*100:.2f}%")
    print(f"  F1-Score:          {f1*100:.2f}%")
    print(f"  False Alarm Rate:  {far*100:.2f}%")
    print(f"  Miss Rate:         {mr*100:.2f}%")
    print()
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Fall']))

if __name__ == '__main__':
    main()
