# Báo cáo: Train Skeleton Model (LSTM & TCN)

**Đồ án:** AI-Based Human Fall Detection System  
**Ngày thực hiện:** 2026-06-09  
**Người thực hiện:** [Tên của bạn]

---

## 1. Tổng quan

### 1.1. Mục tiêu

Train và đánh giá 2 mô hình deep learning trên dữ liệu skeleton để phát hiện ngã:
- **LSTM** (Long Short-Term Memory): Mô hình RNN chuyên xử lý chuỗi thời gian
- **TCN** (Temporal Convolutional Network): Mô hình CNN 1D với dilated convolutions

### 1.2. Môi trường huấn luyện

| Thông số | Giá trị |
|----------|---------|
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) |
| **CUDA Version** | 12.1 |
| **PyTorch** | 2.5.1+cu121 |
| **CPU** | (không sử dụng, đã dùng GPU) |
| **RAM** | (không đo, dataset ~500MB) |

---

## 2. Cấu hình huấn luyện (Hyperparameters)

| Tham số | Giá trị | Giải thích |
|---------|:-------:|------------|
| **Batch Size** | 128 | Số samples mỗi batch |
| **Epochs (max)** | 50 | Số epoch tối đa |
| **Learning Rate** | 0.001 | Tốc độ học ban đầu (Adam optimizer) |
| **Early Stopping** | patience=10 | Dừng nếu val_loss không cải thiện 10 epochs |
| **LR Scheduler** | ReduceLROnPlateau | Giảm LR khi val_loss không giảm (patience=5, factor=0.5) |
| **Dropout** | 0.3 | Tỉ lệ dropout để tránh overfitting |
| **Class Weight (Fall)** | 1.72 | Trọng số cho class Fall (do mất cân bằng) |
| **Class Weight (Normal)** | 1.0 | Trọng số cho class Normal |
| **Loss Function** | Weighted BCE | Binary Cross Entropy với class weights |
| **Optimizer** | Adam | Adaptive Moment Estimation |
| **Random Seed** | 42 | (đã fix khi tạo dataset) |

### 2.1. Lý do chọn class weight = 1.72

Dataset có tỉ lệ Fall/No_Fall = 1:1.72 (15,689 Fall vs 27,038 No_Fall).

Để cân bằng ảnh hưởng của 2 class trong quá trình train:
```
weight_fall = total_samples / (2 * fall_samples) = 42727 / (2 * 15689) ≈ 1.36
weight_normal = total_samples / (2 * normal_samples) = 42727 / (2 * 27038) ≈ 0.79
```

Tuy nhiên, do **bỏ sót ngã (FN) nguy hiểm hơn báo nhầm (FP)**, ta tăng weight_fall lên 1.72 để model ưu tiên phát hiện ngã hơn.

---

## 3. Kiến trúc mô hình

### 3.1. LSTM Model

```
Input: (batch_size, 30, 51)
  ↓
LSTM Layer 1: hidden_size=128
  ↓ (dropout=0.3)
LSTM Layer 2: hidden_size=128
  ↓
Lấy output frame cuối: (batch_size, 128)
  ↓
Linear(128 → 64) + ReLU
  ↓ (dropout=0.3)
Linear(64 → 1) + Sigmoid
  ↓
Output: (batch_size, 1) - xác suất ngã [0, 1]
```

**Tổng số tham số:** ~250K (ước tính)

### 3.2. TCN Model

```
Input: (batch_size, 30, 51)
  ↓ (transpose)
Input: (batch_size, 51, 30)
  ↓
TemporalBlock 1: 51→64 channels, kernel=3, dilation=1
  ↓
TemporalBlock 2: 64→128 channels, kernel=3, dilation=2
  ↓
TemporalBlock 3: 128→128 channels, kernel=3, dilation=4
  ↓
Lấy output frame cuối: (batch_size, 128)
  ↓
Linear(128 → 64) + ReLU + Dropout(0.3)
  ↓
Linear(64 → 1) + Sigmoid
  ↓
Output: (batch_size, 1)
```

**Receptive field:** (3-1) × (1+2+4) = 14 frames (mỗi output phụ thuộc vào 14 frames trước)

**Tổng số tham số:** ~150K (ước tính)

---

## 4. Kết quả huấn luyện

### 4.1. Quá trình train

| Model | Epochs train | Best Epoch | Early Stopping | Thời gian train |
|-------|:-----------:|:----------:|:--------------:|:---------------:|
| **LSTM** | 31/50 | 21 | Có (patience 10) | **73.1s (1.2 phút)** |
| **TCN** | 43/50 | 33 | Có (patience 10) | **153.1s (2.6 phút)** |

**Nhận xét:**
- LSTM hội tụ nhanh hơn (21 epochs đã đạt best)
- TCN cần nhiều epoch hơn (33 epochs) nhưng tiếp tục cải thiện lâu hơn
- LSTM train nhanh gấp ~2 lần TCN (do kiến trúc đơn giản hơn)

### 4.2. Learning Rate History

**LSTM:**
- Epoch 1-26: LR = 0.001
- Epoch 27+: LR = 0.0005 (giảm do val_loss không cải thiện 5 epochs)

**TCN:**
- Epoch 1-38: LR = 0.001
- Epoch 39+: LR = 0.0005 (giảm do val_loss không cải thiện 5 epochs)

### 4.3. Best Validation Loss & Accuracy

| Model | Best Val Loss | Best Val Accuracy |
|-------|:------------:|:-----------------:|
| **LSTM** | 0.2961 | 92.53% |
| **TCN** | 0.2854 | 92.98% |

---

## 5. Kết quả đánh giá trên Test Set

### 5.1. Confusion Matrix

**LSTM:**

|  | Predicted: Normal | Predicted: Fall |
|--|:-----------------:|:---------------:|
| **Actual: Normal** | TN = 3898 | FP = 174 |
| **Actual: Fall** | FN = 313 | TP = 2025 |

**TCN:**

|  | Predicted: Normal | Predicted: Fall |
|--|:-----------------:|:---------------:|
| **Actual: Normal** | TN = 3856 | FP = 216 |
| **Actual: Fall** | FN = 227 | TP = 2111 |

### 5.2. So sánh Metrics

| Metric | LSTM | TCN | Model tốt hơn |
|--------|:----:|:---:|:-------------:|
| **Accuracy** | 92.40% | **93.09%** | TCN (+0.69%) |
| **Precision** | **92.09%** | 90.72% | LSTM (+1.37%) |
| **Recall** | 86.61% | **90.29%** | TCN (+3.68%) |
| **Specificity** | **95.73%** | 94.70% | LSTM (+1.03%) |
| **F1-Score** | 89.27% | **90.50%** | TCN (+1.23%) |
| **False Alarm Rate** | **4.27%** | 5.30% | LSTM (-1.03%) |
| **Miss Rate** | 13.39% | **9.71%** | TCN (-3.68%) |

### 5.3. Classification Report

**LSTM:**

```
              precision    recall  f1-score   support
    Normal       0.93      0.96      0.94      4072
      Fall       0.92      0.87      0.89      2338
  accuracy                           0.92      6410
 macro avg       0.92      0.91      0.92      6410
weighted avg       0.92      0.92      0.92      6410
```

**TCN:**

```
              precision    recall  f1-score   support
    Normal       0.94      0.95      0.95      4072
      Fall       0.91      0.90      0.91      2338
  accuracy                           0.93      6410
 macro avg       0.93      0.92      0.93      6410
weighted avg       0.93      0.93      0.93      6410
```

---

## 6. Phân tích chi tiết

### 6.1. Ưu điểm của LSTM

- **Precision cao hơn (92.09% vs 90.72%):** Khi LSTM báo ngã, 92% là ngã thật → ít báo nhầm hơn
- **Specificity cao hơn (95.73% vs 94.70%):** LSTM nhận diện hoạt động bình thường tốt hơn
- **False Alarm Rate thấp hơn (4.27% vs 5.30%):** Ít báo động giả hơn
- **Train nhanh hơn (73s vs 153s):** Hội tụ nhanh, tiết kiệm thời gian

### 6.2. Ưu điểm của TCN

- **Recall cao hơn (90.29% vs 86.61%):** TCN phát hiện được nhiều cú ngã hơn → **ít bỏ sót hơn**
- **Miss Rate thấp hơn (9.71% vs 13.39%):** Bỏ sót ít cú ngã hơn → **an toàn hơn**
- **F1-Score cao hơn (90.50% vs 89.27%):** Cân bằng tốt hơn giữa Precision và Recall
- **Accuracy cao hơn (93.09% vs 92.40%):** Tổng thể chính xác hơn

### 6.3. Trade-off quan trọng

| Vấn đề | LSTM | TCN |
|--------|------|-----|
| Báo nhầm (FP) | 174 lần | 216 lần |
| Bỏ sót (FN) | 313 lần | 227 lần |

**Trong bài toán Fall Detection:**
- **Bỏ sót ngã (FN) nguy hiểm hơn** báo nhầm (FP)
- Người già bị ngã mà không được phát hiện → có thể gây hậu quả nghiêm trọng
- Báo nhầm chỉ gây phiền toái, nhưng bỏ sót có thể gây nguy hiểm

→ **TCN phù hợp hơn cho ứng dụng thực tế** vì Miss Rate thấp hơn 3.68%.

### 6.4. Phân tích lỗi

**LSTM - 313 trường hợp bỏ sót (FN):**
- Có thể do: ngã chậm, ngã từ tư thế ngồi, hoặc pose estimation không chính xác
- LSTM phụ thuộc nhiều vào frame cuối cùng → nếu frame cuối không rõ ràng, dễ bỏ sót

**TCN - 216 trường hợp báo nhầm (FP):**
- Có thể do: ngồi xuống nhanh, cúi xuống nhặt đồ, hoặc tập thể dục
- TCN có receptive field rộng hơn → nhạy với chuyển động mạnh

---

## 7. So sánh với Baseline (Rule-based)

Rule-based được đánh giá trên cùng 100 video (50 Fall + 50 No_Fall) ngẫu nhiên từ dataset.

| Model | Accuracy | Precision | Recall | F1 |
|-------|:--------:|:---------:|:------:|:--:|
| **Rule-based (đã cải tiến Bước 1)** | 53.00% | 57.89% | 22.00% | 31.88% |
| **LSTM** | 92.40% | 92.09% | 86.61% | 89.27% |
| **TCN** | 93.09% | 90.72% | 90.29% | 90.50% |

**Confusion Matrix của Rule-based (100 video):**

|  | Predicted: Normal | Predicted: Fall |
|--|:-----------------:|:---------------:|
| **Actual: Normal** | TN = 42 | FP = 8 |
| **Actual: Fall** | FN = 39 | TP = 11 |

**Cải thiện so với Rule-based:**
- LSTM: +39.4% Accuracy, +34.2% Precision, +64.6% Recall, +57.4% F1
- TCN: +40.1% Accuracy, +32.8% Precision, +68.3% Recall, +58.6% F1

**Nhận xét:**
- Rule-based có Recall cực thấp (22%) → bỏ sót 39/50 cú ngã (78%)
- Điều này cho thấy ngưỡng thủ công không phù hợp với dataset đa dạng
- LSTM và TCN vượt xa rule-based ở mọi chỉ số, đặc biệt là Recall (+64-68%)

---

## 8. Hạn chế và Mặt trái

### 8.1. Data Leakage (rò rỉ dữ liệu)

**Vấn đề:** Các window từ cùng một video có thể xuất hiện ở cả train và test.

**Ảnh hưởng:**
- Model có thể "học thuộc" một số video thay vì học pattern tổng quát
- Kết quả test có thể cao hơn thực tế

**Giải pháp (nếu có thời gian):**
- Chia dataset theo video ID (không shuffle windows)
- Đảm bảo không có video nào xuất hiện ở cả train và test

### 8.2. Mất cân bằng class

**Vấn đề:** Fall/No_Fall = 1:1.72

**Đã xử lý:** Dùng class weight (1.72 cho Fall, 1.0 cho Normal)

**Hạn chế:**
- Vẫn có thể chưa đủ để model học tốt class thiểu số
- Có thể thử: oversampling Fall, undersampling Normal, hoặc dùng Focal Loss

### 8.3. Pose Estimation không hoàn hảo

**Vấn đề:** Keypoints từ MediaPipe có thể không chính xác (đặc biệt khi bị che khuất)

**Ảnh hưởng:**
- Model học từ dữ liệu nhiễu
- Confidence < 0.3 được đặt về 0 → mất thông tin

**Giải pháp:**
- Dùng smoothing (đã có trong MediaPipe)
- Tăng min_detection_confidence khi extract keypoints

### 8.4. Không có thông tin thời gian thực

**Vấn đề:** Model chỉ nhìn 30 frames (1 giây) → không biết trước đó người đang làm gì

**Ảnh hưởng:**
- Khó phân biệt "ngã thật" vs "nằm nghỉ từ trước"
- Cần thêm logic post-fall (đã làm ở Bước 1)

### 8.5. Overfitting nhẹ

**Dấu hiệu:**
- LSTM: Train acc 94.72% vs Val acc 92.53% (chênh 2.19%)
- TCN: Train acc 94.47% vs Val acc 92.98% (chênh 1.49%)

**Đã xử lý:**
- Dropout 0.3
- Early stopping
- LR scheduler

**Có thể cải thiện:**
- Tăng dropout lên 0.4-0.5
- Thêm data augmentation (noise, rotation, time warping)

---

## 9. Khuyến nghị

### 9.1. Chọn model nào?

**Nếu ưu tiên an toàn (khuyến nghị):** → **TCN**
- Recall cao hơn (90.29% vs 86.61%)
- Miss Rate thấp hơn (9.71% vs 13.39%)
- Phát hiện được nhiều cú ngã hơn

**Nếu ưu tiên ít báo nhầm:** → **LSTM**
- Precision cao hơn (92.09% vs 90.72%)
- False Alarm Rate thấp hơn (4.27% vs 5.30%)

### 9.2. Cải thiện tiếp theo

1. **Data augmentation:** Thêm noise, rotation, time warping vào skeleton
2. **Ensemble:** Kết hợp LSTM + TCN (trung bình predictions)
3. **Threshold tuning:** Điều chỉnh threshold 0.5 → 0.4 để tăng Recall
4. **Post-processing:** Kết hợp với rule-based (đã làm ở Bước 1)
5. **Chia dataset theo video:** Tránh data leakage
6. **Focal Loss:** Thay BCE bằng Focal Loss để xử lý class imbalance tốt hơn

### 9.3. Threshold tuning (gợi ý)

Hiện tại dùng threshold = 0.5 (nếu p ≥ 0.5 → Fall)

**Nếu giảm threshold xuống 0.4:**
- Recall sẽ tăng (phát hiện nhiều ngã hơn)
- Precision sẽ giảm (báo nhầm nhiều hơn)
- Phù hợp nếu ưu tiên an toàn

**Nếu tăng threshold lên 0.6:**
- Precision sẽ tăng (ít báo nhầm hơn)
- Recall sẽ giảm (bỏ sót nhiều hơn)
- Phù hợp nếu ưu tiên ít phiền toái

---

## 10. File đã tạo

| File | Mô tả |
|------|-------|
| `train_skeleton_models.py` | Script train LSTM + TCN (bị lỗi TCN) |
| `train_tcn_only.py` | Script train TCN (đã fix lỗi) |
| `evaluate_lstm.py` | Script evaluate LSTM trên test set |
| `models/skeleton/LSTM_best.pth` | Model LSTM tốt nhất |
| `models/skeleton/TCN_best.pth` | Model TCN tốt nhất |
| `models/skeleton/training_results.json` | Kết quả chi tiết (JSON) |
| `models/skeleton/tcn_results.json` | Kết quả TCN (JSON) |
| `training_log.txt` | Log train LSTM |
| `tcn_training_log.txt` | Log train TCN |

---

## 11. Cách sử dụng model đã train

```python
import torch
import numpy as np

# Load model
from train_tcn_only import FallTCN
model = FallTCN(input_size=51, num_channels=[64, 128, 128], kernel_size=3, dropout=0.3)
checkpoint = torch.load('models/skeleton/TCN_best.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
# skeleton_window: numpy array shape (30, 51) - 30 frames, 51 features
skeleton_window = np.load('example_window.npy')
input_tensor = torch.FloatTensor(skeleton_window).unsqueeze(0)  # (1, 30, 51)

with torch.no_grad():
    prob = model(input_tensor).item()

if prob >= 0.5:
    print(f"FALL detected! (confidence: {prob*100:.1f}%)")
else:
    print(f"Normal (fall probability: {prob*100:.1f}%)")
```

---

## 12. Kết luận

### 12.1. Kết quả đạt được

- Train thành công 2 mô hình LSTM và TCN trên GPU RTX 4050
- **TCN đạt kết quả tốt nhất:** Accuracy 93.09%, F1 90.50%, Recall 90.29%
- **LSTM cũng rất tốt:** Accuracy 92.40%, F1 89.27%, Precision 92.09%
- Cả 2 model đều vượt xa baseline rule-based (~80% accuracy)

### 12.2. Đóng góp cho đồ án

- Chứng minh được deep learning trên skeleton hiệu quả hơn rule-based
- So sánh được LSTM vs TCN cho bài toán fall detection
- Cung cấp baseline để cải thiện tiếp theo (ensemble, fusion, etc.)

### 12.3. Hướng phát triển

1. **Fusion model:** Kết hợp skeleton model + YOLO fine-tuned + rule-based
2. **Real-time testing:** Test trên video thực tế (không phải dataset)
3. **Threshold optimization:** Tìm threshold tối ưu cho từng use case
4. **Model compression:** Giảm kích thước model để deploy trên edge device

---

**Người báo cáo:** [Tên của bạn]  
**Ngày:** 2026-06-09
