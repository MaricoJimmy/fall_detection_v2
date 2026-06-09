# AI-Based Human Fall Detection System

## Giới thiệu

Hệ thống phát hiện ngã sử dụng Computer Vision kết hợp Deep Learning, được phát triển qua 3 giai đoạn:

1. **Rule-based (Baseline)**: Phát hiện ngã dựa trên luật thủ công
2. **Skeleton Dataset**: Xây dựng dataset từ keypoints
3. **Deep Learning (LSTM/TCN)**: Train mô hình học sâu trên dữ liệu skeleton

### Công nghệ sử dụng
- **YOLOv8**: Phát hiện người trong khung hình
- **MediaPipe**: Pose estimation (17 keypoints COCO format)
- **LSTM/TCN**: Mô hình deep learning cho chuỗi thời gian
- **PyTorch**: Framework deep learning

## Kết quả đạt được

### So sánh 3 phương pháp

| Model | Accuracy | Precision | Recall | F1 | Thời gian train |
|-------|:--------:|:---------:|:------:|:--:|:---------------:|
| **Rule-based** | 53.00% | 57.89% | 22.00% | 31.88% | - |
| **LSTM** | 92.40% | 92.09% | 86.61% | 89.27% | 73s |
| **TCN** | 93.09% | 90.72% | **90.29%** | **90.50%** | 153s |

**Nhận xét:**
- Deep learning vượt xa rule-based (+64-68% Recall)
- **TCN** là mô hình tốt nhất với Recall 90.29% (bỏ sót chỉ 9.71%)
- Model đã được train trên GPU RTX 4050 với CUDA 12.1

### Dataset
- **Tổng số video**: 6,988 videos (3,140 Fall + 3,848 No_Fall)
- **Tổng số windows**: 42,727 sliding windows (30 frames mỗi window)
- **Phân chia**: Train 70% / Validation 15% / Test 15%
- **Chi tiết**: Xem [DATASET_BUILD_REPORT.md](DATASET_BUILD_REPORT.md)

## Cài đặt

### Yêu cầu hệ thống
- Python 3.11+
- CUDA 12.1 (để train trên GPU)
- GPU NVIDIA (khuyến nghị RTX 4050 trở lên)
- RAM 16GB+

### Cài đặt thư viện

```bash
# Tạo virtual environment
python -m venv venv_311
venv_311\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Cài PyTorch với CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài scikit-learn
pip install scikit-learn
```

## Sử dụng

### 1. Chạy hệ thống Rule-based (Baseline)

```bash
# Chạy với webcam
python main.py

# Chạy với video
python main.py --source path/to/video.mp4

# Chọn model YOLOv8
python main.py --model yolov8s.pt
```

**Phím tắt:**
- `q`: Thoát
- `r`: Reset detector
- `s`: Lưu ảnh
- `SPACE`: Tạm dừng/Tiếp tục

### 2. Test video với mô hình TCN (Deep Learning)

```bash
# Test video cơ bản
.\venv_311\Scripts\python.exe test_video.py --video "path/to/video.mp4"

# Test và lưu video output
.\venv_311\Scripts\python.exe test_video.py --video "path/to/video.mp4" --output "output.mp4"

# Điều chỉnh threshold (mặc định 0.5)
.\venv_311\Scripts\python.exe test_video.py --video "path/to/video.mp4" --threshold 0.4
```

**Output:**
- Tất cả video output và screenshot được lưu trong thư mục `test_outputs/<video_name>/`
- Ví dụ: `test_outputs/sample_2/output_20260109_123456.mp4`

### 3. Đánh giá mô hình Rule-based

```bash
# Đánh giá trên dataset
python evaluate.py --dataset "C:\Download\Do_an\dataset\video_fall" --limit 50
```

### 4. Train mô hình mới (LSTM/TCN)

```bash
# Train cả LSTM và TCN
python train_skeleton_models.py

# Chỉ train TCN (nếu LSTM đã train xong)
python train_tcn_only.py
```

### 5. Xây dựng skeleton dataset

```bash
# Build dataset từ CSV keypoints
python build_skeleton_dataset.py --dataset "C:\Download\Do_an\dataset\video_fall" --output "datasets/skeleton_windows"

# Build với giới hạn (để test nhanh)
python build_skeleton_dataset.py --dataset "path/to/dataset" --limit 100
```

### 6. Phân tích overfitting

```bash
# Kiểm tra overfitting và vẽ learning curves
python check_overfitting.py
```

## Cấu trúc dự án

```
fall_detection_system/
├── main.py                          # Chạy rule-based với webcam/video
├── config.py                        # Cấu hình hệ thống
├── test_video.py                    # Test video với TCN model
├── evaluate.py                      # Đánh giá rule-based
├── train_skeleton_models.py         # Train LSTM + TCN
├── train_tcn_only.py                # Train TCN riêng
├── build_skeleton_dataset.py        # Xây dựng skeleton dataset
├── check_overfitting.py             # Phân tích overfitting
│
├── utils/                           # Các module chính
│   ├── fall_detector.py             # Rule-based detector (đã cải tiến)
│   ├── pose_estimator.py            # MediaPipe pose estimation
│   └── alert_system.py              # Hệ thống cảnh báo
│
├── models/
│   ├── yolov8n.pt                   # YOLO model
│   └── skeleton/                    # Mô hình deep learning
│       ├── LSTM_best.pth            # LSTM model tốt nhất
│       ├── TCN_best.pth             # TCN model tốt nhất
│       ├── training_results.json    # Kết quả train LSTM
│       ├── tcn_results.json         # Kết quả train TCN
│       └── learning_curves.png      # Biểu đồ learning curves
│
├── datasets/
│   └── skeleton_windows/            # Dataset đã xử lý
│       ├── X_train.npy              # (29908, 30, 51)
│       ├── y_train.npy              # (29908,)
│       ├── X_val.npy                # (6409, 30, 51)
│       ├── y_val.npy                # (6409,)
│       ├── X_test.npy               # (6410, 30, 51)
│       └── y_test.npy               # (6410,)
│
├── test_outputs/                    # Output từ test video
│   └── sample_2/
│       ├── output_20260109_123456.mp4
│       └── screenshot_20260109_123500.jpg
│
├── logs/                            # Log files
├── alerts/                          # Ảnh cảnh báo
│
├── README.md                        # File này
├── DATASET_BUILD_REPORT.md          # Báo cáo xây dựng dataset
├── TRAINING_REPORT.md               # Báo cáo training chi tiết
└── requirements.txt                 # Dependencies
```

## Cải tiến Rule-based (Bước 1)

Đã cải tiến `utils/fall_detector.py` với 4 cải tiến chính:

### 1. Normalize velocity
Chia vận tốc cho chiều cao bounding box để không phụ thuộc khoảng cách camera:
```python
vy_norm = vy / bbox_height
```

### 2. Temporal voting
Xét 15 frame gần nhất, cần ít nhất 10 frame nghi ngờ mới báo ngã:
```python
'temporal_window_size': 15,
'temporal_vote_threshold': 10,
```

### 3. Post-fall confirmation
Kiểm tra người có vẫn nằm sau khi ngã (1 giây):
```python
'post_fall_time': 1.0,
```

### 4. Recovery logic
Tự động reset khi người đứng dậy (30 frames liên tiếp):
```python
'recovery_angle': 30,
'recovery_frames': 30,
```

## Mô hình Deep Learning (Bước 3)

### LSTM (Long Short-Term Memory)
- **Kiến trúc**: 2 lớp LSTM (hidden_size=128) + Fully Connected
- **Input**: (batch_size, 30, 51) - 30 frames, 51 features
- **Ưu điểm**: Precision cao (92.09%), ít báo nhầm
- **Nhược điểm**: Recall thấp hơn TCN (86.61%)

### TCN (Temporal Convolutional Network)
- **Kiến trúc**: 3 TemporalBlocks với dilated convolutions
- **Input**: (batch_size, 30, 51)
- **Ưu điểm**: Recall cao (90.29%), bỏ sót ít hơn
- **Nhược điểm**: False alarm rate cao hơn LSTM

### Hyperparameters
- **Batch size**: 128
- **Learning rate**: 0.001 (Adam optimizer)
- **Epochs**: 50 (early stopping patience=10)
- **Dropout**: 0.3
- **Class weight**: Fall=1.72, Normal=1.0 (xử lý mất cân bằng)
- **Loss function**: Weighted Binary Cross Entropy

## Skeleton Dataset

### Cấu trúc dữ liệu
- **17 keypoints** (COCO format): Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles
- **Mỗi keypoint**: (X, Y, Confidence)
- **Window size**: 30 frames
- **Shape**: (30, 17, 3) → flatten → (30, 51)

### Chuẩn hóa skeleton
1. **Lấy hip center làm gốc**: Trừ tất cả keypoints cho hip center
2. **Chia cho body size**: Khoảng cách shoulder-hip
3. **Xử lý missing keypoints**: Confidence < 0.3 → đặt về 0

### Sliding windows
- **Stride**: 10 frames (overlap)
- **Công thức**: `num_windows = (N - 30) / 10 + 1`
- **Ví dụ**: Video 100 frames → 8 windows

## Hạn chế và Hướng phát triển

### Hạn chế hiện tại
1. **Data leakage**: Các window từ cùng video có thể xuất hiện ở cả train và test
2. **Mất cân bằng class**: Fall/No_Fall = 1:1.72
3. **Overfitting nhẹ**: Loss gap ~0.2, Acc gap ~3%
4. **Phụ thuộc MediaPipe**: Nếu pose estimation sai → model sai

### Hướng phát triển
1. **Chia dataset theo video ID**: Tránh data leakage
2. **Data augmentation**: Thêm noise, rotation, time warping
3. **Ensemble models**: Kết hợp LSTM + TCN
4. **Fusion**: Kết hợp skeleton model + YOLO fine-tuned + rule-based
5. **Model compression**: Giảm kích thước để deploy trên edge device
6. **Threshold tuning**: Tìm threshold tối ưu cho từng use case

## Tài liệu tham khảo

- [DATASET_BUILD_REPORT.md](DATASET_BUILD_REPORT.md) - Chi tiết xây dựng dataset
- [TRAINING_REPORT.md](TRAINING_REPORT.md) - Phân tích kết quả training
- [README_CAI_THIEN_DO_CHINH_XAC.md](README_CAI_THIEN_DO_CHINH_XAC.md) - Hướng dẫn cải thiện độ chính xác

## Tác giả

**Đồ án tốt nghiệp ngành Công nghệ Thông tin**  
**Đề tài**: AI-Based Human Fall Detection System Using Computer Vision  
**Ngày hoàn thành**: 2026-01-09

## License

MIT License
