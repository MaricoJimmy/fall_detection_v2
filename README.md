# Hệ Thống Phát Hiện Ngã (Fall Detection System)

## Giới Thiệu

Hệ thống phát hiện ngã sử dụng **YOLOv8-pose** để phát hiện người và trích xuất 17 keypoints COCO, kết hợp với mô hình **TCN (Temporal Convolutional Network)** để phân loại hành vi té ngã.

### Pipeline chính (TCN)

```
Input Video → YOLOv8-pose → Normalize skeleton → TCN Model → Fall/No Fall
```

## Cài Đặt

### Yêu cầu
- Python 3.10+
- PyTorch (có CUDA nếu dùng GPU)
- Webcam hoặc file video

### Cài dependencies

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # GPU
# hoặc: pip install torch torchvision  # CPU
```

## Cấu Trúc Thư Mục

```
fall_detection_system/
├── main.py                       # Rule-based real-time (webcam/video)
├── test_video.py                 # Test TCN trên video
├── config.py                     # Cấu hình hệ thống
├── evaluate.py                   # Unified evaluator (6 modes)
├── train_tcn_only.py             # Huấn luyện TCN
├── train_skeleton_models.py      # Huấn luyện LSTM + TCN
├── finetune_tcn.py               # Fine-tune TCN trên UR Fall Detection
├── build_skeleton_dataset.py     # Build dataset từ CSV keypoints
├── build_dataset_yolo.py         # Build dataset từ YOLOv8-pose
├── analyze_video_detailed.py     # Phân tích chi tiết từng frame
├── analyze_skeleton_comparison.py# So sánh skeleton
├── check_overfitting.py          # Kiểm tra overfitting + vẽ biểu đồ
│
├── utils/
│   ├── pose_estimator.py         # YOLOv8-pose (thay MediaPipe)
│   ├── fall_detector.py          # Rule-based fall detection
│   └── alert_system.py           # Cảnh báo âm thanh/hình ảnh/log
│
├── models/skeleton/
│   ├── TCN_best.pth              # TCN pre-trained (MediaPipe data)
│   ├── TCN_finetuned.pth         # TCN fine-tuned (UR Fall Detection)
│   ├── LSTM_best.pth             # LSTM pre-trained
│   ├── learning_curves.png       # Biểu đồ learning curves
│   └── gap_analysis.png          # Biểu đồ gap analysis
│
└── datasets/skeleton_windows/
    ├── X_train.npy, y_train.npy
    ├── X_val.npy, y_val.npy
    └── X_test.npy, y_test.npy
```

## Sử Dụng

### Chạy real-time (rule-based)

```bash
python main.py                         # Webcam
python main.py --source video.mp4      # File video
```

### Test TCN trên video

```bash
python test_video.py --video path/to/video.mp4
```

### Đánh giá

```bash
# TCN+YOLOv8-pose trên thư mục video
python evaluate.py --mode tcn --base path/to/video_folder

# LSTM trên test set
python evaluate.py --mode lstm

# Phân tích ngưỡng
python evaluate.py --mode thresholds

# So sánh pre/post fine-tune
python evaluate.py --mode subject4 --base path/to/Subject4
```

### Huấn luyện

```bash
python train_tcn_only.py               # Train TCN từ đầu
python finetune_tcn.py                 # Fine-tune TCN
```

### Phân tích

```bash
python check_overfitting.py            # Vẽ learning curves + gap analysis
python analyze_video_detailed.py --video path/to/video.mp4
```

## Kết Quả Đánh Giá

### So sánh các phương pháp

| Phương pháp | Dataset | Accuracy | Recall | F1 | False Alarm |
|-------------|---------|----------|--------|-----|-------------|
| Rule-based (cũ) | videoevaluate (60 video) | 56.67% | 16.13% | 27.78% | 0.0% |
| TCN + YOLOv8m + MediaPipe | videoevaluate (60 video) | 60.0% | 87.1% | 69.23% | 69.0% |
| TCN + YOLOv8-Pose (motion=0.25) | videoevaluate (80 video) | 67.50% | 87.80% | 73.47% | 53.85% |
| **TCN Fine-tuned (motion=0)** | **Subject 4 (37 video)** | **72.97%** | **100%** | **77.27%** | **50.00%** |

### TCN + YOLOv8m trên videoevaluate (60 video Le2i)

| Metric | Value |
|--------|-------|
| Accuracy | 60.0% |
| Precision | 57.45% |
| **Recall** | **87.1%** |
| Specificity | 31.03% |
| F1 | 69.23% |

### Ảnh hưởng của Threshold (TCN + YOLOv8m)

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| 0.50 | 60.0% | 57.45% | 87.1% | 69.23 |
| **0.75** | **60.0%** | **58.97%** | **74.19%** | **65.71** |
| 0.85 | 60.0% | 60.0% | 67.74% | 63.64 |

### Fine-tune trên UR Fall Detection (Subject 4, motion=0)

| Model | TP | FP | TN | FN | Accuracy | Recall | F1 | False Alarm |
|-------|----|----|----|----|----------|--------|-----|-------------|
| Pre-fine-tune | 16 | 13 | 7 | 1 | 62.16% | 94.12% | 69.57% | 65.00% |
| **Fine-tuned** | **17** | **10** | **10** | **0** | **72.97%** | **100%** | **77.27%** | **50.00%** |

### Vấn đề còn tồn tại

- ⚠️ False alarm còn cao (~50%): Model chưa phân biệt được ngã vs nằm xuống/ngủ
- 🔧 Cần thêm velocity features để phân biệt ngã (tốc độ cao) vs nằm (tốc độ thấp)
- 🔧 Cần train lại TCN từ đầu với YOLOv8-pose keypoints (bỏ qua pre-trained MediaPipe)

## Keypoints (COCO 17 điểm)

```
 0: Nose            1: Left Eye        2: Right Eye
 3: Left Ear        4: Right Ear       5: Left Shoulder
 6: Right Shoulder  7: Left Elbow      8: Right Elbow
 9: Left Wrist     10: Right Wrist    11: Left Hip
12: Right Hip      13: Left Knee      14: Right Knee
15: Left Ankle     16: Right Ankle
```
