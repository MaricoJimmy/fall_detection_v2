"""
Script train LSTM va TCN cho Fall Detection
Hien thi progress bar, ghi log chi tiet, luu model tot nhat
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# === CAU HINH ===
CONFIG = {
    'data_dir': 'datasets/skeleton_windows',
    'output_dir': 'models/skeleton',
    'batch_size': 128,
    'epochs': 50,
    'learning_rate': 0.001,
    'patience': 10,  # Early stopping patience
    'window_size': 30,
    'num_features': 51,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# === LSTM MODEL ===
class FallLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.classifier(last_output)

# === TCN MODEL ===
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.padding = padding
    
    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Crop ve dung kich thuoc input
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Crop ve dung kich thuoc input
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class FallTCN(nn.Module):
    def __init__(self, input_size=51, num_channels=[64, 128, 128], kernel_size=3, dropout=0.3):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation_size = 2 ** i
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq, features) -> (batch, features, seq)
        x = self.network(x)
        x = x[:, :, -1]  # Lay output cuoi
        return self.classifier(x)

# === TRAINING FUNCTION ===
def train_model(model, model_name, train_loader, val_loader, config):
    print(f"\n{'='*70}")
    print(f"  TRAINING {model_name}")
    print(f"{'='*70}\n")
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Class weights de xu ly mat can bang
    weight_fall = torch.tensor([1.72]).to(device)
    weight_normal = torch.tensor([1.0]).to(device)
    
    criterion = nn.BCELoss(weight=None)  # Binary Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # === TRAINING PHASE ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Weighted loss
            weights = torch.where(y_batch == 1, weight_fall, weight_normal)
            loss = nn.functional.binary_cross_entropy(outputs, y_batch, weight=weights)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # === VALIDATION PHASE ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).float().unsqueeze(1)
                
                outputs = model(X_batch)
                weights = torch.where(y_batch == 1, weight_fall, weight_normal)
                loss = nn.functional.binary_cross_entropy(outputs, y_batch, weight=weights)
                
                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Luu history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # === HIEN THI PROGRESS ===
        print(f"Epoch [{epoch+1:3d}/{config['epochs']}] "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
              f"| LR: {current_lr:.6f} "
              f"| Time: {epoch_time:.1f}s", end='')
        
        # === EARLY STOPPING & SAVE BEST MODEL ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Luu model tot nhat
            save_path = os.path.join(config['output_dir'], f'{model_name}_best.pth')
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
            print(f" [patience: {patience_counter}/{config['patience']}]", end='')
        
        print()  # New line
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {config['patience']} epochs)")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE - {model_name}")
    print(f"{'='*70}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val acc: {history['val_acc'][best_epoch-1]:.4f}")
    
    return model, history, total_time, best_epoch

# === EVALUATION FUNCTION ===
def evaluate_model(model, test_loader, model_name, config):
    print(f"\n{'='*70}")
    print(f"  EVALUATING {model_name} ON TEST SET")
    print(f"{'='*70}\n")
    
    device = torch.device(config['device'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(predicted.flatten())
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Tinh metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Confusion Matrix:")
    print(f"  TN: {tn:6d}  FP: {fp:6d}")
    print(f"  FN: {fn:6d}  TP: {tp:6d}")
    print()
    print(f"Metrics:")
    print(f"  Accuracy:          {accuracy*100:.2f}%")
    print(f"  Precision:         {precision*100:.2f}%")
    print(f"  Recall:            {recall*100:.2f}%")
    print(f"  Specificity:       {specificity*100:.2f}%")
    print(f"  F1-Score:          {f1*100:.2f}%")
    print(f"  False Alarm Rate:  {false_alarm_rate*100:.2f}%")
    print(f"  Miss Rate:         {miss_rate*100:.2f}%")
    
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Fall']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'false_alarm_rate': false_alarm_rate,
        'miss_rate': miss_rate,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    }

# === MAIN FUNCTION ===
def main():
    print("="*70)
    print("  FALL DETECTION - SKELETON MODEL TRAINING")
    print("="*70)
    print(f"Device: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print("="*70)
    
    # === LOAD DATA ===
    print("\n[1/5] Loading dataset...")
    X_train = np.load(os.path.join(CONFIG['data_dir'], 'X_train.npy'))
    y_train = np.load(os.path.join(CONFIG['data_dir'], 'y_train.npy'))
    X_val = np.load(os.path.join(CONFIG['data_dir'], 'X_val.npy'))
    y_val = np.load(os.path.join(CONFIG['data_dir'], 'y_val.npy'))
    X_test = np.load(os.path.join(CONFIG['data_dir'], 'X_test.npy'))
    y_test = np.load(os.path.join(CONFIG['data_dir'], 'y_test.npy'))
    
    print(f"  Train: {X_train.shape} - Fall: {np.sum(y_train==1)}, Normal: {np.sum(y_train==0)}")
    print(f"  Val:   {X_val.shape} - Fall: {np.sum(y_val==1)}, Normal: {np.sum(y_val==0)}")
    print(f"  Test:  {X_test.shape} - Fall: {np.sum(y_test==1)}, Normal: {np.sum(y_test==0)}")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # === CREATE OUTPUT DIR ===
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # === TRAIN LSTM ===
    print("\n[2/5] Training LSTM model...")
    lstm_model = FallLSTM(
        input_size=CONFIG['num_features'],
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    lstm_model, lstm_history, lstm_time, lstm_best_epoch = train_model(
        lstm_model, 'LSTM', train_loader, val_loader, CONFIG
    )
    
    # === TRAIN TCN ===
    print("\n[3/5] Training TCN model...")
    tcn_model = FallTCN(
        input_size=CONFIG['num_features'],
        num_channels=[64, 128, 128],
        kernel_size=3,
        dropout=0.3
    )
    tcn_model, tcn_history, tcn_time, tcn_best_epoch = train_model(
        tcn_model, 'TCN', train_loader, val_loader, CONFIG
    )
    
    # === EVALUATE MODELS ===
    print("\n[4/5] Evaluating models on test set...")
    
    # Load best models
    lstm_checkpoint = torch.load(os.path.join(CONFIG['output_dir'], 'LSTM_best.pth'))
    lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
    lstm_metrics = evaluate_model(lstm_model, test_loader, 'LSTM', CONFIG)
    
    tcn_checkpoint = torch.load(os.path.join(CONFIG['output_dir'], 'TCN_best.pth'))
    tcn_model.load_state_dict(tcn_checkpoint['model_state_dict'])
    tcn_metrics = evaluate_model(tcn_model, test_loader, 'TCN', CONFIG)
    
    # === SAVE RESULTS ===
    print("\n[5/5] Saving results...")
    
    results = {
        'config': CONFIG,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'train_fall': int(np.sum(y_train.numpy() == 1)),
            'train_normal': int(np.sum(y_train.numpy() == 0)),
            'val_fall': int(np.sum(y_val.numpy() == 1)),
            'val_normal': int(np.sum(y_val.numpy() == 0)),
            'test_fall': int(np.sum(y_test.numpy() == 1)),
            'test_normal': int(np.sum(y_test.numpy() == 0)),
        },
        'LSTM': {
            'training_time': lstm_time,
            'best_epoch': lstm_best_epoch,
            'history': lstm_history,
            'test_metrics': lstm_metrics
        },
        'TCN': {
            'training_time': tcn_time,
            'best_epoch': tcn_best_epoch,
            'history': tcn_history,
            'test_metrics': tcn_metrics
        }
    }
    
    # Save JSON
    with open(os.path.join(CONFIG['output_dir'], 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {CONFIG['output_dir']}/training_results.json")
    print(f"Models saved to: {CONFIG['output_dir']}/LSTM_best.pth and TCN_best.pth")
    
    # === FINAL COMPARISON ===
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time':<10}")
    print("-"*70)
    print(f"{'LSTM':<10} {lstm_metrics['accuracy']*100:<10.2f} {lstm_metrics['precision']*100:<10.2f} "
          f"{lstm_metrics['recall']*100:<10.2f} {lstm_metrics['f1']*100:<10.2f} {lstm_time:<10.1f}s")
    print(f"{'TCN':<10} {tcn_metrics['accuracy']*100:<10.2f} {tcn_metrics['precision']*100:<10.2f} "
          f"{tcn_metrics['recall']*100:<10.2f} {tcn_metrics['f1']*100:<10.2f} {tcn_time:<10.1f}s")
    print("="*70)

if __name__ == '__main__':
    main()
