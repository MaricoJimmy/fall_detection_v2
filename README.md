# Hệ Thống Phát Hiện Ngã (Fall Detection System)

## Giới Thiệu

Hệ thống phát hiện ngã sử dụng YOLOv8-pose để phát hiện người và trích xuất keypoints, kết hợp với mô hình TCN (Temporal Convolutional Network) để phân loại hành vi té ngã theo thời gian thực.

### Kiến trúc

```
Input Video → YOLOv8-pose (detection + pose) → Normalize skeleton → TCN Model → Fall/No Fall
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
├── main.py                       # Hệ thống chính (rule-based + webcam)
├── test_video.py                 # Test TCN trên video
├── config.py                     # Cấu hình hệ thống
│
├── train_tcn_only.py             # Huấn luyện mô hình TCN
├── train_skeleton_models.py      # Huấn luyện LSTM + TCN
├── finetune_tcn.py               # Fine-tune TCN
│
├── build_skeleton_dataset.py     # Xây dựng dataset từ CSV keypoints
├── build_dataset_yolo.py         # Xây dựng dataset từ YOLOv8-pose
│
├── evaluate_pose_tcn.py          # Đánh giá TCN + YOLOv8-pose
├── evaluate_tcn_yolo.py          # Đánh giá TCN batch
├── eval_thresholds.py            # Tunning ngưỡng phát hiện
├── eval_subject4.py              # So sánh pre vs fine-tuned
│
├── analyze_video_detailed.py     # Phân tích chi tiết từng frame
├── analyze_skeleton_comparison.py# So sánh skeleton
├── check_overfitting.py          # Kiểm tra overfitting
│
├── utils/
│   ├── pose_estimator.py         # Pose estimation (MediaPipe)
│   ├── fall_detector.py          # Phát hiện ngã rule-based
│   └── alert_system.py           # Hệ thống cảnh báo
│
├── models/skeleton/              # Mô hình đã huấn luyện
│   ├── TCN_best.pth              # TCN tốt nhất
│   ├── LSTM_best.pth             # LSTM tốt nhất
│   └── TCN_finetuned.pth         # TCN fine-tuned
│
└── datasets/skeleton_windows/    # Dataset dạng numpy
    ├── X_train.npy, y_train.npy
    ├── X_val.npy, y_val.npy
    └── X_test.npy, y_test.npy
```

## Sử Dụng

### Chạy thử trên video

```bash
python test_video.py --video path/to/video.mp4
```

### Huấn luyện TCN từ đầu

```bash
python train_tcn_only.py
```

### Fine-tune TCN với dữ liệu YOLOv8-pose

```bash
python finetune_tcn.py
```

### Đánh giá trên tập video

```bash
python evaluate_pose_tcn.py --base path/to/video_folder
```

## Mô Hình

### TCN (Temporal Convolutional Network)
- **Input**: 30 frames x 51 features (17 keypoints x 3)
- **Architecture**: 3 TemporalBlocks (channels [64, 128, 128], kernel_size=3)
- **Output**: Xác suất ngã (0-1)

### Kết quả đánh giá

| Metric | Value |
|--------|-------|
| Accuracy | 93.09% |
| Precision | 90.72% |
| Recall | 90.29% |
| F1-Score | 90.50% |

## Các Phase Phát Triển

1. **Phase 1**: Rule-based (YOLOv8 detect + MediaPipe pose + angle/velocity heuristics)
2. **Phase 2**: Xây dựng skeleton dataset, sliding windows normalization
3. **Phase 3**: Deep learning (LSTM → TCN) với TCN đạt kết quả tốt nhất

## Keypoints

Sử dụng 17 keypoints theo chuẩn COCO:

```
0: Nose          1: Left Eye       2: Right Eye
3: Left Ear      4: Right Ear      5: Left Shoulder
6: Right Shoulder 7: Left Elbow    8: Right Elbow
9: Left Wrist   10: Right Wrist   11: Left Hip
12: Right Hip   13: Left Knee     14: Right Knee
15: Left Ankle  16: Right Ankle
```
