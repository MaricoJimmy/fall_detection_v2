# Báo Cáo Tổng Hợp Hành Trình Phát Triển Hệ Thống Phát Hiện Ngã

## Mục Lục
1. [Giai Đoạn 1: Rule-based Baseline](#giai-doan-1-rule-based-baseline)
2. [Giai Đoạn 2: LSTM](#giai-doan-2-lstm)
3. [Giai Đoạn 3: TCN](#giai-doan-3-tcn)
4. [Giai Đoạn 4: Nâng Cấp YOLO](#giai-doan-4-nang-cap-yolo)
5. [Tổng Kết](#tong-ket)

---

## Giai Đoạn 1: Rule-based Baseline

### Thuật toán
- Sử dụng MediaPipe Pose trích xuất 17 keypoints
- Phát hiện ngã dựa trên 3 tiêu chí:
  1. **Góc nghiêng cơ thể**: >55°
  2. **Tỷ lệ khung hình (w/h)**: >1.2
  3. **Vận tốc rơi dọc**: >100 px/frame

### Các cải tiến đã áp dụng
1. Chuẩn hóa vận tốc theo chiều cao bounding box
2. Temporal voting (15 frame, cần ≥10 frame nghi ngờ)
3. Post-fall confirmation (nằm ≥1s mới confirm)
4. Recovery detection (tự động reset khi đứng dậy)

### Kết quả (trên tập validation cũ - 6988 video)
- **Độ chính xác (Accuracy)**: 78.17%
- **Độ nhạy (Recall)**: 22.0%
- **Độ đặc hiệu (Specificity)**: 99.0%

### Kết quả (trên tập videoevaluate mới - 60 video, 31F/29NF)
- **Accuracy**: 56.67%
- **Precision**: 100.0%
- **Recall**: 16.13%
- **Specificity**: 100.0%
- **F1**: 27.78%
- **Nhận xét**: Chỉ bắt được 5/31 ca ngã thật nhưng không có false alarm nào. Quá bảo thủ.

---

## Giai Đoạn 2: LSTM

### Kiến trúc
- LSTM 2 lớp (hidden_size=128)
- Input: 51 features (17 keypoints × 3 tọa độ)
- Window: 30 frames
- Output: Fall probability (sigmoid)

### Huấn luyện
- Dataset: 6988 video (từ CSV chứ không phải từ raw video gốc)
- Train/Val/Test split
- Optimizer: Adam, LR: 0.001
- Batch size: 64, Epochs: 50
- Early stopping

### Kết quả (trên tập validation cũ - data leakage)
- **Validation Accuracy**: ~80%
- **Test Accuracy**: ~82%

### Kết quả (trên tập videoevaluate mới - 60 video)
- **Accuracy**: 38.33%
- **Precision**: 41.67%
- **Recall**: 48.39%
- **Specificity**: 27.59%
- **F1**: 44.78%
- **Nhận xét**: Hiệu suất kém nhất trong các phương pháp, nhiều false alarm.

---

## Giai Đoạn 3: TCN

### Kiến trúc
- Temporal Convolutional Network
- Channels: [64, 128, 128]
- Kernel size: 3
- Dropout: 0.3
- Input: (batch, 30, 51) → Output: (batch, 1)

### So sánh với LSTM
- TCN nhanh hơn LSTM do có thể parallel trong quá trình train
- TCN capture temporal patterns tốt hơn với dilated convolutions
- TCN không bị vanishing gradient như LSTM

### Kết quả (trên tập validation cũ - data leakage)
- **Test Accuracy**: 93.09%
- **Precision**: 93.56%
- **Recall**: 92.59%
- **F1-Score**: 93.07%
- **AUC-ROC**: 97.46%

### Kết quả TCN + YOLOv8n (trên tập videoevaluate mới - 60 video)
- **Accuracy**: 46.67%
- **Precision**: 48.65%
- **Recall**: 58.06%
- **Specificity**: 34.48%
- **F1**: 52.94%

### Kết quả TCN + YOLOv8m (trên tập videoevaluate mới - 60 video)
- **Accuracy**: 60.0%
- **Precision**: 57.45%
- **Recall**: 87.1%
- **Specificity**: 31.03%
- **F1**: 69.23%
- **Threshold**: 0.5
- **Nhận xét**: Recall cao nhất trong các phương pháp (87.1%). YOLOv8m cải thiện đáng kể so với YOLOv8n.

### Cải thiện specificity bằng cách tăng threshold (TCN + YOLOv8m)

| Threshold | Accuracy | Precision | Recall | Specificity | F1 | TP | FP |
|-----------|----------|-----------|--------|-------------|-----|----|----|
| 0.50 | 60.0% | 57.45% | 87.1% | 31.03% | 69.23 | 27 | 20 |
| **0.75** | **60.0%** | **58.97%** | **74.19%** | **44.83%** | **65.71** | **23** | **16** |
| 0.85 | 60.0% | 60.0% | 67.74% | 51.72% | 63.64 | 21 | 14 |

**Chọn threshold = 0.75**: recall giảm từ 87% → 74% (miss thêm 4 ca) nhưng specificity tăng từ 31% → 45% (giảm 4 false alarm). Giá trị cân bằng nhất giữa bắt đúng ngã và hạn chế báo nhầm.

---

## Giai Đoạn 4: Nâng Cấp YOLO

### Thay đổi
- Từ YOLOv8n (nano) → YOLOv8m (medium)
- Lý do: Phát hiện người chính xác hơn ở khoảng cách xa và góc máy khó

### Kết quả thực tế trên videoevaluate (60 video Le2i)
- **TP**: 27/31 (detect đúng 27/31 ca ngã thật)
- **FP**: 20 (báo động giả trên 29 video No_Fall)
- **Recall**: 87.1% - cải thiện rõ rệt so với YOLOv8n (58%)
- **F1**: 69.23% - tốt nhất trong tất cả phương pháp
- **Accuracy**: 60.0%

---

## Giai Đoạn 5: Chuyển Từ MediaPipe Sang YOLOv8-Pose + Motion Filter

### Lý do
- **MediaPipe yếu ở góc trên xuống (top-down):** Model gốc của MediaPipe train trên ảnh trực diện, không detect được pose từ góc camera gắn trần.
- **Cần crop person region:** MediaPipe xử lý trên region đã crop, mất context ảnh, giảm độ chính xác.
- **Không hỗ trợ nhiều người:** Code cũ chỉ lấy detection đầu tiên.
- **Chuyển sang YOLOv8-pose:** 1 forward pass cho cả bbox + 17 keypoints COCO, không cần crop, hỗ trợ nhiều người.

### Vấn đề mới phát sinh
Khi chuyển từ MediaPipe → YOLOv8-pose, keypoints confidence khác nhau nên TCN model (train trên data từ MediaPipe) bị **domain shift**. Kết quả trên 80 video mới:

| Metric | Chưa có motion filter | Có motion filter (t=0.25) |
|--------|----------------------|--------------------------|
| TP | 41 | 36 |
| FP | 33 | 21 |
| TN | 6 | 18 |
| FN | 0 | 5 |
| Accuracy | 58.75% | **67.50%** |
| Precision | 55.41% | **63.16%** |
| Recall | 100.00% | **87.80%** |
| Specificity | 10.26% | **46.15%** |
| F1 | 71.30% | **73.47%** |
| False Alarm | 89.74% | **53.85%** |

### Kết quả với các motion threshold khác nhau (threshold TCN=0.75)

| Motion Threshold | TP | FP | TN | FN | Accuracy | F1 | Recall | False Alarm | Miss Rate |
|-----------------|----|----|----|----|----------|----|--------|-------------|-----------|
| 0.50 | 22 | 10 | 29 | 19 | 63.75% | 60.27% | 53.66% | 25.64% | 46.34% |
| 0.30 | 32 | 17 | 22 | 9 | 67.50% | 71.11% | 78.05% | 43.59% | 21.95% |
| **0.25** | **36** | **21** | **18** | **5** | **67.50%** | **73.47%** | **87.80%** | **53.85%** | **12.20%** |
| 0.20 | 41 | 28 | 11 | 0 | 65.00% | 74.55% | 100.00% | 71.79% | 0.00% |

**Chọn motion_threshold=0.25**: Cân bằng tốt nhất giữa Recall (87.8%) và False Alarm (53.85%).

### Nguyên nhân false alarm còn cao
- **No_Fall có người nằm/ngủ:** Skeleton nằm ngang giống hệt fall, TCN predict Fall với confidence rất cao (~1.0).
- **No_Fall có chuyển động mạnh:** Một số video No_Fall có motion lớn (chạy nhảy) khiến motion filter không lọc được.
- **Ngược lại:** Một số Fall video có motion thấp (ngã từ từ, ngã khi đã nằm sẵn) bị motion filter chặn.

### Hướng khắc phục
1. **Thu thập thêm No_Fall có người nằm/ngủ** để retrain TCN model.
2. **Thêm velocity features** vào input (tính tốc độ khung xương giữa các frame) để TCN phân biệt ngã (tốc độ cao) vs nằm sẵn (tốc độ thấp).
3. **Dùng YOLOv8-pose extract keypoints** cho `build_skeleton_dataset.py` để rebuild dataset, sau đó train lại TCN/LSTM từ đầu.

### Tác động
- **Pipeline mới:** YOLOv8-pose → Normalize → Motion buffer → TCN predict
- **Tốc độ:** Chậm hơn MediaPipe ~1.5x nhưng đơn giản hơn (không cần crop, không cần 2 model riêng)
- **File thay đổi:** `test_video.py` (đã chuyển), `evaluate_pose_tcn.py` (mới), `baocaotonghop.md` (đang cập nhật)
- **Yêu cầu:** `yolov8m-pose.pt` (~83MB, tự download lần chạy đầu)

---

## Giai Đoạn 6: Fine-tune TCN trên UR Fall Detection (datafinetune)

### Dữ liệu fine-tune
| Subject | Fall | ADL (No_Fall) | Nguồn |
|---------|------|---------------|-------|
| Subject 1 | 16 | 16 | UR Fall Detection (góc chéo, camera trần) |
| Subject 2 | 25 | 23 | UR Fall Detection |
| Subject 3 | 21 | 22 | UR Fall Detection |
| **Subject 4** | **17** | **20** | **UR Fall Detection (dùng để test)** |
| **Total** | **79** | **81** | **160 video** |

- ADL gồm nhiều hoạt động **nằm/ngủ**, **tập thể dục dưới sàn**, **cúi nhặt đồ** → chính xác là các trường hợp gây false alarm trước đây.
- Video ngắn (2-17s), ~30fps, trích xuất được **4351 skeleton windows** (mỗi window 30 frame).
- Train (80%) / Val (20%) split, shuffle.

### Quá trình fine-tune
- Model: `FallTCN` pre-trained trên MediaPipe data (giữ nguyên kiến trúc)
- Learning rate: 3e-4 (thấp hơn train từ đầu)
- Epochs: 30, Early stopping patience: 8
- Loss: BCELoss + class weight
- **Kết quả trên validation set (datafinetune):**
  - Best val accuracy: **95.63%** (epoch 27)
  - Best val loss: **0.2579**

### Kết quả trên Subject 4 (cross-dataset test - 17F/20NF)

#### Không dùng motion filter (motion_threshold=0)

| Model | TP | FP | TN | FN | Accuracy | Recall | F1 | False Alarm |
|-------|----|----|----|----|----------|--------|-----|-------------|
| **Pre-fine-tune** | 16 | 13 | 7 | 1 | 62.16% | 94.12% | 69.57% | 65.00% |
| **Fine-tuned** | **17** | **10** | **10** | **0** | **72.97%** | **100%** | **77.27%** | **50.00%** |

- ✅ **Catch thêm 1 Fall** (16→17, đạt 100% recall)
- ✅ **Giảm 3 False Alarm** (13→10)
- ✅ **F1 tăng 7.7%** (69.57%→77.27%)

#### Có motion filter (motion_threshold=0.25)

| Model | TP | FP | TN | FN | Accuracy | Recall | F1 | False Alarm |
|-------|----|----|----|----|----------|--------|-----|-------------|
| Pre-fine-tune | 8 | 2 | 18 | 9 | 70.27% | 47.06% | 59.26% | 10.00% |
| **Fine-tuned** | **10** | **2** | **18** | **7** | **75.68%** | **58.82%** | **68.97%** | **10.00%** |

- ✅ **Catch thêm 2 Fall** (8→10)
- Motion filter 0.25 vẫn quá aggressive (miss 7/17 Fall)

### Nhận xét
1. **Fine-tune cải thiện rõ rệt** trên cross-dataset Subject 4: F1 +7.7%, FA -15%, Recall đạt 100%.
2. **Motion filter không phù hợp** với UR Fall Detection vì nhiều Fall video có chuyển động chậm (ngã từ từ, ngã trên giường).
3. **False alarm còn 50%** do 10/20 No_Fall vẫn bị predict sai (chủ yếu là nằm/ngủ).

### Hướng tiếp theo
1. **Thêm velocity features** vào input: (30, 51) → (30, 102) để TCN phân biệt ngã (tốc độ cao) vs nằm xuống (tốc độ thấp).
2. **Thu thập thêm No_Fall** có nằm/ngủ từ nhiều góc camera khác nhau.
3. **Train từ đầu** với velocity features + YOLOv8-pose keypoints (bỏ qua pre-trained MediaPipe model).

---

## Tổng Kết

### So sánh các phương pháp

| Phương pháp | Dataset | Accuracy | Recall | F1 | False Alarm |
|-------------|---------|----------|--------|-----|-------------|
| Rule-based (cũ) | videoevaluate (60 video) | 56.67% | 16.13% | 27.78% | 0.0% |
| TCN + YOLOv8m + MediaPipe | videoevaluate (60 video) | 60.0% | 87.1% | 69.23% | 69.0% |
| TCN + YOLOv8-Pose (motion=0.25) | videoevaluate (80 video) | 67.50% | 87.80% | 73.47% | 53.85% |
| **TCN Fine-tuned (motion=0)** | **Subject 4 (37 video)** | **72.97%** | **100%** | **77.27%** | **50.00%** |

### Vấn đề còn tồn tại
1. ✅ ~~Data leakage + Domain shift~~: **Đã fix bằng fine-tune trên YOLOv8-pose keypoints.**
2. ⚠️ **False alarm vẫn cao** (50%): Model chưa phân biệt được ngã vs nằm xuống/ngủ.
3. ✅ ~~Motion filter quá aggressive~~: **Fine-tuned model không cần motion filter vẫn đạt recall 100% trên Subject 4.**
4. 🔧 **Cần thêm velocity features** để phân biệt ngã (tốc độ cao) vs nằm (tốc độ thấp).

### Hướng phát triển
1. ✅ Rule-based baseline
2. ✅ LSTM-based detection
3. ✅ TCN-based detection
4. ✅ Evaluation trên Le2i với YOLOv8m vs YOLOv8n
5. ✅ Chuyển từ MediaPipe → YOLOv8-pose
6. ✅ **Fine-tune TCN trên UR Fall Detection (datafinetune)**
7. 🔧 **Thêm velocity features + train lại TCN từ đầu**
8. 🔧 **Đánh giá fine-tuned model trên videoevaluate2 (80 video cũ)**
9. ❌ Triển khai real-time (nếu kết quả khả quan)
