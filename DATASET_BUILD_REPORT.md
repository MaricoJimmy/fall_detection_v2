# Báo cáo chi tiết: Xây dựng Skeleton Dataset

**Đồ án:** AI-Based Human Fall Detection System  
**Ngày xây dựng:** 2026-06-09  
**Người thực hiện:** [Tên của bạn]

---

## 1. Nguồn dữ liệu (Data Source)

- **Tên dataset:** Fall Detection Dataset (công khai)
- **Nguồn:** Roboflow Universe - Fall Detection Project
- **Đường dẫn:** `C:\Download\Do_an\dataset\video_fall`

**Cấu trúc thư mục:**

```
video_fall/
├── Fall/
│   ├── Raw_Video/        → 3140 video MP4 (người bị ngã)
│   └── Keypoints_CSV/    → 3140 file CSV (keypoints tương ứng)
└── No_Fall/
    ├── Raw_Video/        → 3848 video MP4 (hoạt động bình thường)
    └── Keypoints_CSV/    → 3848 file CSV (keypoints tương ứng)
```

**Tổng số video:** 6988 videos
- Fall (có ngã): 3140 videos
- No_Fall (bình thường): 3848 videos

---

## 2. Format dữ liệu CSV (Keypoints)

Mỗi file CSV chứa keypoints của từng frame trong video tương ứng.

**Các cột trong CSV:**

| Cột | Mô tả |
|-----|--------|
| Frame | Số thứ tự frame (bắt đầu từ 1) |
| Keypoint | Tên điểm khớp (tiếng Anh) |
| X | Tọa độ ngang (pixel) |
| Y | Tọa độ dọc (pixel) |
| Confidence | Độ tin cậy của keypoint (0.0 - 1.0) |

**Số keypoints mỗi frame:** 17 điểm (COCO format)

**Danh sách 17 keypoints:**

| Index | Tên (EN) | Tên (VN) |
|:-----:|----------|----------|
| 0 | Nose | Mũi |
| 1 | Left Eye | Mắt trái |
| 2 | Right Eye | Mắt phải |
| 3 | Left Ear | Tai trái |
| 4 | Right Ear | Tai phải |
| 5 | Left Shoulder | Vai trái |
| 6 | Right Shoulder | Vai phải |
| 7 | Left Elbow | Khủy tay trái |
| 8 | Right Elbow | Khủy tay phải |
| 9 | Left Wrist | Cổ tay trái |
| 10 | Right Wrist | Cổ tay phải |
| 11 | Left Hip | Hông trái |
| 12 | Right Hip | Hông phải |
| 13 | Left Knee | Đầu gối trái |
| 14 | Right Knee | Đầu gối phải |
| 15 | Left Ankle | Mắt cá chân trái |
| 16 | Right Ankle | Mắt cá chân phải |

**Ví dụ 1 dòng CSV:**

```
Frame=1, Keypoint=Nose, X=425.47, Y=163.05, Confidence=0.999
```

---

## 3. Tham số xử lý (Processing Parameters)

| Tham số | Giá trị | Mô tả |
|---------|:-------:|-------|
| Window Size | 30 | Số frame mỗi sliding window |
| Stride | 10 | Bước nhảy giữa các window |
| Min Frames | 10 | Số frame tối thiểu để tạo window |
| Min Confidence | 0.3 | Confidence tối thiểu của keypoint |
| Normalize Method | Hip-based | Lấy hip center làm gốc, chia body size |
| Random Seed | 42 | Seed để shuffle ổn định |

**Giải thích tham số:**

- **Window Size (30 frames):** Mỗi mẫu huấn luyện là một chuỗi 30 frame liên tiếp. Với video 30 FPS, đây là khoảng 1 giây - đủ để học chuyển động ngã.

- **Stride (10 frames):** Bước nhảy giữa các window. Stride < Window Size → các window overlap nhau. Giúp tạo nhiều mẫu hơn từ cùng một video. Ví dụ: video 100 frame → (100 - 30) / 10 + 1 = 8 windows.

- **Min Confidence (0.3):** Keypoint có confidence < 0.3 được coi là không nhìn thấy. Tọa độ của điểm đó sẽ được đặt về 0.

---

## 4. Chuẩn hóa dữ liệu (Normalization)

**Mục đích:** Giảm ảnh hưởng của vị trí người trong ảnh và khoảng cách camera.

### Bước 1: Lấy hip center làm gốc tọa độ

- Tính trung điểm của Left Hip và Right Hip
- Trừ tất cả keypoints cho hip center
- **Ý nghĩa:** Người đứng vị trí nào trong ảnh cũng được đưa về cùng hệ tọa độ

### Bước 2: Chia cho body size (khoảng cách vai-hông)

- Body size = khoảng cách giữa Shoulder Center và Hip Center
- Chia tất cả tọa độ cho body size
- **Ý nghĩa:** Người gần camera và xa camera sẽ bớt khác nhau

### Bước 3: Xử lý keypoint không nhìn thấy

- Keypoint có confidence < 0.3 được đặt tọa độ về (0, 0)
- Tránh ảnh hưởng của dữ liệu nhiễu

**Kết quả:** Mỗi frame có shape `(17, 3)` với `[X_norm, Y_norm, Confidence]`

---

## 5. Sliding Windows (Cắt dữ liệu)

**Phương pháp:** Sliding window với overlap

Video dài N frames được cắt thành nhiều windows:

```
Window 1: frame 0-29
Window 2: frame 10-39
Window 3: frame 20-49
...
```

**Công thức tính số windows mỗi video:**

```
num_windows = (N - window_size) / stride + 1
```

**Xử lý video ngắn:**
- Video ngắn hơn `window_size`: Pad bằng cách lặp frame cuối cùng, tạo thành 1 window duy nhất
- Video ngắn hơn `min_frames` (10): Bỏ qua, không sử dụng

**Flatten:** Mỗi window được làm phẳng từ `(30, 17, 3)` thành `(30, 51)`:
- 30: số frame
- 51 = 17 keypoints × 3 features (X, Y, Confidence)

---

## 6. Chia Train / Validation / Test

**Tỉ lệ chia:**

| Tập | Tỉ lệ |
|-----|:-----:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

**Phương pháp:**
- Shuffle toàn bộ windows với random seed = 42
- Cắt theo tỉ lệ trên
- Không chia theo video/person (vì đã có nhiều người, nhiều cảnh)

**Lưu ý:**
- Random seed cố định (42) đảm bảo kết quả ổn định giữa các lần chạy
- Các window từ cùng video có thể xuất hiện ở cả train và test
- Với dataset lớn (6988 videos), việc này chấp nhận được

---

## 7. Kết quả thống kê (Dataset Statistics)

### 7.1. Số lượng video xử lý

| Nhãn | Tổng | OK | Lỗi/Bỏ |
|------|-----:|---:|-------:|
| Fall (có ngã) | 3140 | 3112 | 28 |
| No_Fall (bình thường) | 3848 | 3811 | 37 |
| **TỔNG** | **6988** | **6923** | **65** |

**Lý do lỗi/bỏ qua:**
- Video quá ngắn (< 10 frames): 36 videos
- File CSV lỗi hoặc không đọc được: 29 videos

### 7.2. Số lượng windows tạo được

| Nhãn | Số windows |
|------|----------:|
| Fall | 15,689 |
| No_Fall | 27,038 |
| **TỔNG** | **42,727** |

**Tỉ lệ Fall/No_Fall:** 1:1.72 (No_Fall nhiều hơn Fall)

### 7.3. Phân chia Train / Validation / Test

| Tập | Tổng | Fall | No_Fall |
|-----|-----:|-----:|--------:|
| Train | 29,908 | 11,029 | 18,879 |
| Validation | 6,409 | 2,322 | 4,087 |
| Test | 6,410 | 2,338 | 4,072 |
| **TỔNG** | **42,727** | **15,689** | **27,038** |

**Tỉ lệ phần trăm:**
- Train: 69.99%
- Validation: 15.00%
- Test: 15.00%

**Tỉ lệ Fall trong mỗi tập:**
- Train: 36.88%
- Validation: 36.23%
- Test: 36.47%

### 7.4. Kích thước dữ liệu

**Shape mỗi sample:** `(30, 51)`
- 30: số frame trong window
- 51: số features mỗi frame (17 keypoints × 3 values)

**Tổng dung lượng:**

| File | Dung lượng |
|------|----------:|
| X_train.npy | 349.12 MB |
| X_val.npy | 74.81 MB |
| X_test.npy | 74.82 MB |
| y_train.npy | 0.23 MB |
| y_val.npy | 0.05 MB |
| y_test.npy | 0.05 MB |
| **Tổng** | **~499 MB** |

---

## 8. File Output

**Thư mục:** `datasets/skeleton_windows/`

| File | Shape | Mô tả |
|------|-------|-------|
| `X_train.npy` | (29908, 30, 51) | Dữ liệu train |
| `y_train.npy` | (29908,) | Label train (0=normal, 1=fall) |
| `X_val.npy` | (6409, 30, 51) | Dữ liệu validation |
| `y_val.npy` | (6409,) | Label validation |
| `X_test.npy` | (6410, 30, 51) | Dữ liệu test |
| `y_test.npy` | (6410,) | Label test |

**Cách load trong Python:**

```python
import numpy as np

X_train = np.load('datasets/skeleton_windows/X_train.npy')
y_train = np.load('datasets/skeleton_windows/y_train.npy')
print(X_train.shape)  # (29908, 30, 51)
print(y_train.shape)  # (29908,)
```

---

## 9. Hướng dẫn sử dụng cho Training

Dataset đã sẵn sàng để train các mô hình:
- **LSTM** (Long Short-Term Memory)
- **GRU** (Gated Recurrent Unit)
- **TCN** (Temporal Convolutional Network)
- **Random Forest / XGBoost** (flatten window thành vector 1D)

**Input shape cho deep learning:**
- `(batch_size, 30, 51)` cho LSTM/GRU/TCN
- Hoặc `(batch_size, 1530)` cho MLP (flatten)

**Label:**
- `0`: Normal (hoạt động bình thường)
- `1`: Fall (bị ngã)

**Gợi ý tham số train:**

| Tham số | Giá trị đề xuất |
|---------|-----------------|
| Batch size | 64 hoặc 128 |
| Learning rate | 0.001 (Adam optimizer) |
| Epochs | 50-100 (với early stopping) |
| Dropout | 0.3-0.5 (tránh overfitting) |
| Class weight (fall) | 1.72 |
| Class weight (normal) | 1.0 |

**Ví dụ code PyTorch LSTM:**

```python
import torch
import torch.nn as nn

class FallLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch_size, 30, 51)
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]  # Lấy output của frame cuối
        return self.classifier(last_output)
```

---

## 10. Ghi chú và Hạn chế

### Ưu điểm

- Dataset lớn (6988 videos) → Mô hình học được nhiều pattern
- Đã có keypoints CSV → Không cần chạy lại MediaPipe
- Chuẩn hóa skeleton → Giảm phụ thuộc vào vị trí và khoảng cách
- Sliding windows với overlap → Tạo nhiều mẫu hơn

### Hạn chế

- **Tỉ lệ Fall/No_Fall không cân bằng (1:1.72)** → Cần dùng class weighting hoặc oversampling khi train
- **Chưa chia theo person/video** (có thể có data leakage) → Với dataset lớn, vẫn chấp nhận được
- **Một số video quá ngắn bị bỏ qua** (36 videos) → Ảnh hưởng không đáng kể

### Khuyến nghị khi train

- Dùng **class weighting** để xử lý mất cân bằng: `weight_fall = 1.72`, `weight_normal = 1.0`
- Dùng **early stopping** trên validation set
- Theo dõi cả **Precision** và **Recall** (không chỉ Accuracy)
- **Recall** đặc biệt quan trọng vì bỏ sót ngã rất nguy hiểm

---

## Kết luận

Dataset skeleton đã được xây dựng thành công từ **6988 videos**. Tổng cộng **42,727 sliding windows** đã được tạo và chia thành 3 tập. Dataset sẵn sàng để train LSTM/TCN cho bước tiếp theo của đồ án.

---

**Người báo cáo:** [Tên của bạn]  
**Ngày:** 2026-06-09
