"""
Fine-tune TCN model on YOLOv8-pose keypoints from datafinetune
Loads pre-trained TCN (trained on MediaPipe data), fine-tunes on new data
"""
import os, sys, json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_tcn_only import FallTCN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRETRAINED_PATH = 'models/skeleton/TCN_best.pth'
DATA_DIR = 'datasets/skeleton_windows'
OUTPUT_DIR = 'models/skeleton'

CONFIG = {
    'batch_size': 64,
    'epochs': 30,
    'learning_rate': 3e-4,
    'patience': 8,
    'val_split': 0.2,
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    X = np.load(os.path.join(DATA_DIR, 'X_finetune.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y_finetune.npy'))
    print(f"Data: X {X.shape}, Fall {int(y.sum())} / ADL {len(y)-int(y.sum())}")
    n_val = int(len(X) * CONFIG['val_split'])
    indices = np.random.RandomState(42).permutation(len(X))
    X, y = X[indices], y[indices]
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=CONFIG['batch_size'], shuffle=False)
    print(f"\nLoading pre-trained TCN from {PRETRAINED_PATH} ...")
    model = FallTCN(input_size=51, num_channels=[64, 128, 128], kernel_size=3, dropout=0.3)
    ckpt = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    weight_fall = torch.tensor([1 - y_train.mean(), 1.0])[1] / y_train.mean()
    criterion = nn.BCELoss(weight=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            weights = torch.where(y_batch == 1, weight_fall, torch.tensor(1.0).to(DEVICE))
            loss = nn.functional.binary_cross_entropy(outputs, y_batch, weight=weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).float().unsqueeze(1)
                outputs = model(X_batch)
                weights = torch.where(y_batch == 1, weight_fall, torch.tensor(1.0).to(DEVICE))
                loss = nn.functional.binary_cross_entropy(outputs, y_batch, weight=weights)
                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch [{epoch+1:2d}/{CONFIG['epochs']}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}", end='')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            save_path = os.path.join(OUTPUT_DIR, 'TCN_finetuned.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, save_path)
            print(" [BEST]", end='')
        else:
            patience_counter += 1
            print(f" [{patience_counter}/{CONFIG['patience']}]", end='')
        print()
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    print(f"\nBest epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    results = {
        'config': CONFIG,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'history': history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }
    with open(os.path.join(OUTPUT_DIR, 'finetune_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved fine-tuned model to {save_path}")
    print(f"Done!")

if __name__ == '__main__':
    import sys
    main()
