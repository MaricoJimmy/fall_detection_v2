# HƯỚNG DẪN CHẠY DEMO

## Bước 1: Cài đặt thư viện

Mở terminal/cmd tại thư mục `fall_detection_system` và chạy:

```bash
pip install -r requirements.txt
```

Hoặc chạy file batch:

```bash
run_setup.bat
```

## Bước 2: Kiểm tra hệ thống

```bash
python test_system.py
```

## Bước 3: Chạy Demo

### Demo nhanh (Recommend cho lần đầu)

```bash
python demo_quick_start.py
```

### Demo đầy đủ với tất cả tính năng

```bash
python main.py
```

## Bước 4: Sử dụng Demo

### Demo nhanh (demo_quick_start.py)
- Tự động mở webcam
- Phát hiện người bằng YOLOv8
- Pose estimation với MediaPipe
- Vẽ skeleton trên cơ thể
- Phát hiện ngã dựa trên:
  - Góc nghiêng cơ thể > 45 độ
  - Tỷ lệ width/height > 1.2
  - Thời gian ngã > 1 giây
- Cảnh báo âm thanh khi ngã
- Phím 'q' để thoát, 'r' để reset

### Demo đầy đủ (main.py)
- Tất cả tính năng của demo nhanh
- Bảng thông tin chi tiết
- Lưu ảnh khi ngã
- Log file sự kiện
- FPS counter
- Thanh trạng thái
- Phím tắt:
  - q: Thoát
  - r: Reset detector
  - s: Lưu ảnh
  - SPACE: Pause/Resume

## Test với Webcam

1. Đứng trước camera ở vị trí nhìn thấy toàn thân
2. Di chuyển bình thường để xem skeleton
3. Thực hiện động tác ngã giả lập:
   - Nghiêng cơ thể mạnh
   - Nằm xuống đất
   - Quay sang ngang

## Test với File Video

```bash
python main.py --source video_test.mp4
```

Video test có thể tải từ:
- UR Fall Detection Dataset
- Le2i Fall Detection Dataset
- Tự quay video test

## Thay đổi Model YOLO

```bash
# Nano - nhanh nhất
python main.py --model yolov8n.pt

# Small - cân bằng
python main.py --model yolov8s.pt

# Medium - chính xác hơn
python main.py --model yolov8m.pt
```

## Cấu hình nâng cao

Mở file `config.py` để thay đổi:

### Ngưỡng phát hiện ngã
```python
FALL_DETECTION_CONFIG = {
    'angle_threshold': 45,  # Góc nghiêng (độ)
    'aspect_ratio_threshold': 1.2,  # Tỷ lệ W/H
    'fall_time_threshold': 1.0,  # Thời gian xác nhận
}
```

### Camera
```python
CAMERA_CONFIG = {
    'source': 0,  # Webcam index
    'width': 640,
    'height': 480,
}
```

## Lưu ý

- Camera cần đặt ở vị trí nhìn thấy toàn thân người
- Ánh sáng tốt sẽ tăng độ chính xác
- Model YOLOv8n nhanh nhưng có thể bỏ sót người ở xa
- Cân nhắc dùng yolov8s.pt hoặc yolov8m.pt để tăng độ chính xác

## Troubleshooting

### Camera không mở
- Kiểm tra camera đang không bị sử dụng bởi app khác
- Thử thay đổi `CAMERA_CONFIG['source']` = 1 hoặc 2

### YOLOv8 không load
- Model sẽ tự download lần đầu tiên (~6MB)
- Kiểm tra kết nối internet

### MediaPipe lỗi
- Cài đặt lại: pip install mediapipe --upgrade

### FPS thấp
- Giảm độ phân giải trong config.py
- Dùng model YOLOv8n thay vì yolov8m

## Support

Nếu gặp vấn đề, kiểm tra:
1. File test_system.py
2. Log trong thư mục logs/
3. README.md