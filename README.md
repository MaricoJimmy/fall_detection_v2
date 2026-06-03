# AI-Based Human Fall Detection System

## Giới thiệu

Hệ thống phát hiện ngã của con người sử dụng Computer Vision với sự kết hợp của:
- **YOLOv8**: Phát hiện người trong khung hình
- **MediaPipe**: Pose estimation để xác định vị trí các điểm khớp
- **Pose-based Fall Detection**: Thuật toán phát hiện ngã dựa trên vị trí và chuyển động

## Tính năng

### 1. Phát hiện người (YOLOv8)
- Sử dụng mô hình YOLOv8 pretrained
- Phát hiện người thời gian thực với độ chính xác cao
- Tự động crop vùng chứa người để xử lý

### 2. Pose Estimation (MediaPipe)
- Xác định 33 điểm khớp trên cơ thể
- Theo dõi chuyển động thời gian thực
- Vẽ skeleton lên cơ thể người

### 3. Phát hiện ngã
Sử dụng nhiều tiêu chí:
- **Góc nghiêng cơ thể**: Góc giữa trục cơ thể và phương thẳng đứng
- **Tỷ lệ Width/Height**: Bounding box thay đổi khi ngã
- **Vận tốc**: Phát hiện chuyển động nhanh xuống dưới
- **Vị trí điểm khớp**: Kiểm tra vị trí tương đối của các khớp

### 4. Hệ thống cảnh báo
- **Cảnh báo trực quan**: Viền màu và text cảnh báo trên màn hình
- **Cảnh báo âm thanh**: Beep sound khi phát hiện ngã
- **Lưu ảnh**: Tự động lưu ảnh khi có sự cố
- **Log file**: Ghi nhận tất cả sự kiện

### 5. Giao diện
- Hiển thị thời gian thực
- Bảng thông tin với các chỉ số
- FPS counter
- Thanh trạng thái

## Cài đặt

### Yêu cầu
- Python 3.8+
- OpenCV
- Webcam hoặc file video

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Sử dụng

### Chạy với webcam (mặc định)

```bash
python main.py
```

### Chạy với file video

```bash
python main.py --source path/to/video.mp4
```

### Chọn model YOLOv8 khác

```bash
# Model nhỏ, nhanh
python main.py --model yolov8n.pt

# Model medium, cân bằng
python main.py --model yolov8m.pt

# Model lớn, chính xác hơn
python main.py --model yolov8l.pt
```

## Phím tắt

| Phím | Chức năng |
|------|-----------|
| `q` | Thoát chương trình |
| `r` | Reset detector |
| `s` | Lưu ảnh hiện tại |
| `SPACE` | Tạm dừng/Tiếp tục |

## Cấu trúc dự án

```
fall_detection_system/
├── main.py                 # File chính
├── config.py              # Cấu hình hệ thống
├── requirements.txt       # Thư viện cần thiết
├── README.md              # Hướng dẫn sử dụng
├── utils/
│   ├── __init__.py
│   ├── fall_detector.py   # Thuật toán phát hiện ngã
│   ├── pose_estimator.py  # Pose estimation
│   └── alert_system.py    # Hệ thống cảnh báo
├── models/                # Lưu model YOLOv8
├── logs/                  # Log file
└── alerts/                # Ảnh cảnh báo
```

## Tùy chỉnh

### Thay đổi ngưỡng phát hiện

Trong file `config.py`:

```python
FALL_DETECTION_CONFIG = {
    'angle_threshold': 45,      # Góc nghiêng (độ)
    'aspect_ratio_threshold': 1.2,  # Tỷ lệ W/H
    'velocity_threshold': 15,   # Vận tốc
    'fall_time_threshold': 1.0, # Thời gian xác nhận ngã (giây)
}
```

### Thay đổi nguồn camera

```python
CAMERA_CONFIG = {
    'source': 0,  # 0, 1, 2... cho webcam, hoặc đường dẫn file video
    'width': 640,
    'height': 480,
    'fps': 30
}
```

## Thuật toán phát hiện ngã

### 1. Tính góc nghiêng cơ thể

```
- Tính trung điểm của vai và hông
- Vector trục cơ thể = vai_center - hông_center
- Tính góc giữa vector trục cơ thể và vector thẳng đứng (0, -1)
- Nếu góc > threshold (45°) -> khả năng ngã
```

### 2. Tính tỷ lệ Width/Height

```
- Khi người đứng: height > width, aspect_ratio < 1
- Khi người nằm: width > height, aspect_ratio > 1
- Nếu aspect_ratio > 1.2 -> khả năng ngã
```

### 3. Kiểm tra vận tốc

```
- Theo dõi vị trí trung tâm của bounding box
- Tính vận tốc di chuyển (pixels/giây)
- Vận tốc dọc lớn xuống dưới -> khả năng ngã
```

### 4. Kiểm tra vị trí điểm khớp

```
- Kiểm tra: mắt cá chân cao hơn hông
- Kiểm tra: độ sâu z của các điểm khớp
- Phát hiện tư thế bất thường
```

### 5. Logic kết hợp

```
fall_score = số chỉ số cho thấy đang ngã
- Nếu fall_score >= 2 trong thời gian >= 1 giây -> Xác nhận ngã
```

## Demo Video

Để test hệ thống, bạn có thể:
1. Sử dụng webcam và thực hiện các động tác ngã giả lập
2. Tải video test từ các nguồn như:
   - UR Fall Detection Dataset
   - Le2i Fall Detection Dataset
   - YouTube videos về fall detection

## Ghi chú

- Hệ thống hoạt động tốt nhất với camera đặt ở vị trí có thể nhìn thấy toàn thân
- Ánh sáng tốt giúp tăng độ chính xác
- Model YOLOv8n nhanh nhưng có thể bỏ sót người ở xa, cân nhắc dùng yolov8s hoặc yolov8m

## Tác giả

Đồ án tốt nghiệp ngành Công nghệ Thông tin
Đề tài: AI-Based Human Fall Detection System Using Computer Vision

## License

MIT License