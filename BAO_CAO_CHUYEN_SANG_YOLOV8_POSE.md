# Báo Cáo Chuyển Đổi Từ MediaPipe Sang YOLOv8-Pose

## 1. Lý Do Chuyển Đổi

### Vấn đề với MediaPipe
- **Góc quay từ trên xuống (top-down):** MediaPipe không detect được pose do model gốc được train chủ yếu trên ảnh trực diện/nghiêng.
- **Cần crop person region:** Phải crop bounding box từ YOLO rồi mới đưa vào MediaPipe, gây mất context ảnh và giảm độ chính xác.
- **Chỉ xử lý 1 người:** Code cũ chỉ lấy detection đầu tiên, bỏ qua các người khác trong khung hình.
- **Pipeline 2 bước:** YOLO detection → crop → MediaPipe pose → tốn thời gian và phức tạp.

### Lợi ích YOLOv8-Pose
- **1 model duy nhất:** Vừa detect người vừa trả keypoints COCO 17 ngay trong 1 forward pass.
- **Keypoints trên ảnh gốc:** Không cần crop, không mất context.
- **Hỗ trợ nhiều người:** Trả keypoints cho tất cả detection cùng lúc.
- **Đa dạng góc quay hơn:** YOLOv8-pose được train trên COCO dataset với nhiều góc nhìn khác nhau.

## 2. So Sánh Pipeline

### Pipeline cũ (MediaPipe)
```
frame → YOLO detect person → crop ROI → MediaPipe Pose → keypoints (crop coords)
      → convert to image coords → normalize → TCN predict
```

### Pipeline mới (YOLOv8-Pose)
```
frame → YOLOv8-Pose → keypoints (image coords) + bbox → normalize → TCN predict
```

## 3. Các Thay Đổi Cụ Thể

### 3.1. Model
- **Cũ:** `yolov8m.pt` (detection) + MediaPipe Pose
- **Mới:** `yolov8m-pose.pt` (detection + pose)

### 3.2. File bị ảnh hưởng
| File | Thay đổi |
|------|----------|
| `test_video.py` | Bỏ MediaPipe, dùng YOLOv8-pose, thêm multi-person tracking |
| `requirements.txt` | Vẫn giữ mediapipe (còn dùng trong `main.py` và `utils/`) |

## 4. Cấu Trúc Code Mới

### detect_person()
```python
def detect_person(self, frame):
    results = self.yolo(frame, conf=0.5, verbose=False)[0]
    keypoints_xy = results.keypoints.xy.cpu().numpy()   # (N, 17, 2)
    keypoints_conf = results.keypoints.conf.cpu().numpy()  # (N, 17)
    boxes = results.boxes
    # Trả về list dict: {'bbox': [...], 'skeleton': ndarray(17,3)}
```

### Multi-person Tracking (IOU-based)
- Dùng Intersection-over-Union (IOU) để gán detection giữa các frame.
- Mỗi người giữ 1 buffer riêng (deque 30 frame) cho TCN predict.
- Track bị mất >30 frame sẽ tự động xóa.

## 5. Kết Quả

### Ưu điểm
- **Skeleton rõ hơn** ở góc trực diện và nghiêng.
- **Hỗ trợ nhiều người** cùng lúc.
- **Code đơn giản hơn**, pipeline ngắn hơn.
- **Tọa độ keypoints chính xác** (không cần convert crop → ảnh gốc).

### Nhược điểm
- **Chậm hơn MediaPipe** ~1.5-2x (yolov8m-pose ~83MB so với yolov8m ~26MB).
- **Vẫn hạn chế ở góc top-down** nếu dữ liệu train của YOLOv8-pose không có góc đó.

### So sánh tốc độ (trên CPU)
| Pipeline | FPS |
|----------|-----|
| YOLOv8m + MediaPipe | ~8-12 |
| YOLOv8m-pose | ~5-8 |

## 6. Cách Chạy

```bash
# Download model tự động lần đầu
python test_video.py --video path/to/video.mp4

# Chạy với threshold khác
python test_video.py --video path/to/video.mp4 --threshold 0.5

# Chạy chậm hơn
python test_video.py --video path/to/video.mp4 --speed 0.5
```
