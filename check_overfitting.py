"""
Script kiem tra overfitting bang cach phan tich tu training logs
"""
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
import io

# Set stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def parse_training_log(log_file):
    """Parse training log de lay history"""
    # Thu cac encoding khac nhau
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252']
    content = None
    
    for encoding in encodings:
        try:
            with open(log_file, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if content is None:
        raise ValueError(f"Cannot read {log_file} with any encoding")
    
    # Tim cac epoch lines
    pattern = r'Epoch \[\s*(\d+)/\d+\].*Train Loss: ([\d.]+) Acc: ([\d.]+).*Val Loss: ([\d.]+) Acc: ([\d.]+)'
    matches = re.findall(pattern, content)
    
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    for match in matches:
        epoch, t_loss, t_acc, v_loss, v_acc = match
        epochs.append(int(epoch))
        train_loss.append(float(t_loss))
        train_acc.append(float(t_acc))
        val_loss.append(float(v_loss))
        val_acc.append(float(v_acc))
    
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }

def analyze_overfitting(history, model_name):
    """Phan tich overfitting tu history"""
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    print(f"\n{'='*70}")
    print(f"  OVERFITTING ANALYSIS - {model_name}")
    print(f"{'='*70}\n")
    
    # Tinh gap o epoch cuoi
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    
    loss_gap = final_train_loss - final_val_loss
    acc_gap = final_train_acc - final_val_acc
    
    # Tim best epoch
    best_val_loss = min(val_loss)
    best_epoch = val_loss.index(best_val_loss) + 1
    
    print(f"Training Summary:")
    print(f"  Total epochs: {len(train_loss)}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print()
    
    print(f"Final Metrics (Epoch {len(train_loss)}):")
    print(f"  Train Loss: {final_train_loss:.4f}")
    print(f"  Val Loss:   {final_val_loss:.4f}")
    print(f"  Loss Gap:   {loss_gap:.4f} ({'+' if loss_gap > 0 else ''}{loss_gap*100:.2f}%)")
    print()
    print(f"  Train Acc:  {final_train_acc*100:.2f}%")
    print(f"  Val Acc:    {final_val_acc*100:.2f}%")
    print(f"  Acc Gap:    {acc_gap*100:.2f}%")
    print()
    
    # Phan tich overfitting
    print("Overfitting Indicators:")
    
    # 1. Loss gap
    if abs(loss_gap) < 0.05:
        print(f"  ✓ Loss gap nhỏ ({abs(loss_gap):.4f}) → Không overfitting")
        loss_score = "GOOD"
    elif abs(loss_gap) < 0.15:
        print(f"  ⚠ Loss gap trung bình ({abs(loss_gap):.4f}) → Overfitting nhẹ")
        loss_score = "MODERATE"
    else:
        print(f"  ✗ Loss gap lớn ({abs(loss_gap):.4f}) → Overfitting nghiêm trọng")
        loss_score = "SEVERE"
    
    # 2. Accuracy gap
    if abs(acc_gap) < 0.02:
        print(f"  ✓ Acc gap nhỏ ({acc_gap*100:.2f}%) → Không overfitting")
        acc_score = "GOOD"
    elif abs(acc_gap) < 0.05:
        print(f"  ⚠ Acc gap trung bình ({acc_gap*100:.2f}%) → Overfitting nhẹ")
        acc_score = "MODERATE"
    else:
        print(f"  ✗ Acc gap lớn ({acc_gap*100:.2f}%) → Overfitting nghiêm trọng")
        acc_score = "SEVERE"
    
    # 3. Val loss trend (co tang o cuoi khong)
    last_5_val_loss = val_loss[-5:]
    if last_5_val_loss[-1] > last_5_val_loss[0]:
        print(f"  ⚠ Val loss tăng ở 5 epoch cuối → Dấu hiệu overfitting")
        trend_score = "WARNING"
    else:
        print(f"  ✓ Val loss giảm/ổn định ở 5 epoch cuối → Không overfitting")
        trend_score = "GOOD"
    
    # 4. Train loss co giam lien tuc khong
    train_loss_decreasing = all(train_loss[i] >= train_loss[i+1] for i in range(len(train_loss)-1))
    if train_loss_decreasing:
        print(f"  ✓ Train loss giảm liên tục → Model đang học tốt")
    else:
        print(f"  ⚠ Train loss không giảm liên tục → Có thể có vấn đề")
    
    print()
    
    # Ket luan
    print("Conclusion:")
    if loss_score == "GOOD" and acc_score == "GOOD" and trend_score == "GOOD":
        print("  ✓ KHÔNG OVERFITTING - Model generalize tốt")
        overfit_level = "NONE"
    elif loss_score == "SEVERE" or acc_score == "SEVERE":
        print("  ✗ OVERFITTING NGHIÊM TRỌNG - Cần regularization mạnh hơn")
        overfit_level = "SEVERE"
    else:
        print("  ⚠ OVERFITTING NHẸ - Có thể cải thiện bằng dropout/augmentation")
        overfit_level = "MILD"
    
    return {
        'loss_gap': loss_gap,
        'acc_gap': acc_gap,
        'loss_score': loss_score,
        'acc_score': acc_score,
        'trend_score': trend_score,
        'overfit_level': overfit_level,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }

def plot_learning_curves(lstm_history, tcn_history):
    """Ve learning curves cho ca 2 models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # LSTM Loss
    ax = axes[0, 0]
    ax.plot(lstm_history['epochs'], lstm_history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(lstm_history['epochs'], lstm_history['val_loss'], label='Val Loss', linewidth=2)
    best_epoch = lstm_history['val_loss'].index(min(lstm_history['val_loss'])) + 1
    ax.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('LSTM - Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # LSTM Accuracy
    ax = axes[0, 1]
    ax.plot(lstm_history['epochs'], [x*100 for x in lstm_history['train_acc']], label='Train Acc', linewidth=2)
    ax.plot(lstm_history['epochs'], [x*100 for x in lstm_history['val_acc']], label='Val Acc', linewidth=2)
    ax.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('LSTM - Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TCN Loss
    ax = axes[1, 0]
    ax.plot(tcn_history['epochs'], tcn_history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(tcn_history['epochs'], tcn_history['val_loss'], label='Val Loss', linewidth=2)
    best_epoch = tcn_history['val_loss'].index(min(tcn_history['val_loss'])) + 1
    ax.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('TCN - Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TCN Accuracy
    ax = axes[1, 1]
    ax.plot(tcn_history['epochs'], [x*100 for x in tcn_history['train_acc']], label='Train Acc', linewidth=2)
    ax.plot(tcn_history['epochs'], [x*100 for x in tcn_history['val_acc']], label='Val Acc', linewidth=2)
    ax.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('TCN - Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/skeleton/learning_curves.png', dpi=150, bbox_inches='tight')
    print("\n✓ Đã lưu biểu đồ: models/skeleton/learning_curves.png")
    plt.close()

def plot_gap_analysis(lstm_history, tcn_history):
    """Ve gap giua train va val qua cac epoch"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LSTM Gap
    ax = axes[0]
    lstm_loss_gap = [t - v for t, v in zip(lstm_history['train_loss'], lstm_history['val_loss'])]
    ax.plot(lstm_history['epochs'], lstm_loss_gap, label='Loss Gap (Train - Val)', linewidth=2, color='blue')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap')
    ax.set_title('LSTM - Train vs Val Gap (Loss)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TCN Gap
    ax = axes[1]
    tcn_loss_gap = [t - v for t, v in zip(tcn_history['train_loss'], tcn_history['val_loss'])]
    ax.plot(tcn_history['epochs'], tcn_loss_gap, label='Loss Gap (Train - Val)', linewidth=2, color='orange')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap')
    ax.set_title('TCN - Train vs Val Gap (Loss)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/skeleton/gap_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Đã lưu biểu đồ: models/skeleton/gap_analysis.png")
    plt.close()

def main():
    print("="*70)
    print("  OVERFITTING ANALYSIS")
    print("="*70)
    
    # Parse logs
    print("\nParsing training logs...")
    lstm_history = parse_training_log('training_log.txt')
    tcn_history = parse_training_log('tcn_training_log.txt')
    
    print(f"  LSTM: {len(lstm_history['epochs'])} epochs")
    print(f"  TCN: {len(tcn_history['epochs'])} epochs")
    
    # Analyze
    lstm_analysis = analyze_overfitting(lstm_history, 'LSTM')
    tcn_analysis = analyze_overfitting(tcn_history, 'TCN')
    
    # Plot
    print(f"\n{'='*70}")
    print("  PLOTTING LEARNING CURVES")
    print(f"{'='*70}")
    
    plot_learning_curves(lstm_history, tcn_history)
    plot_gap_analysis(lstm_history, tcn_history)
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'Loss Gap':<12} {'Acc Gap':<12} {'Overfit Level':<15}")
    print("-"*70)
    print(f"{'LSTM':<10} {lstm_analysis['loss_gap']:<12.4f} {lstm_analysis['acc_gap']*100:<12.2f}% {lstm_analysis['overfit_level']:<15}")
    print(f"{'TCN':<10} {tcn_analysis['loss_gap']:<12.4f} {tcn_analysis['acc_gap']*100:<12.2f}% {tcn_analysis['overfit_level']:<15}")
    print("="*70)
    
    # Recommendations
    print("\nRecommendations:")
    if lstm_analysis['overfit_level'] == "NONE" and tcn_analysis['overfit_level'] == "NONE":
        print("  ✓ Cả 2 model đều không overfitting → Có thể sử dụng được")
    elif lstm_analysis['overfit_level'] == "SEVERE" or tcn_analysis['overfit_level'] == "SEVERE":
        print("  ✗ Có model bị overfitting nghiêm trọng:")
        print("    - Tăng dropout (0.3 → 0.4 hoặc 0.5)")
        print("    - Thêm data augmentation")
        print("    - Giảm model complexity (ít layers hơn)")
        print("    - Dùng L2 regularization")
    else:
        print("  ⚠ Có overfitting nhẹ, có thể cải thiện bằng:")
        print("    - Tăng dropout nhẹ (0.3 → 0.35)")
        print("    - Thêm data augmentation (noise, rotation)")
        print("    - Early stopping (đã dùng)")

if __name__ == '__main__':
    main()
