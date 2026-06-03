"""
Cấu hình cho hệ thống phát hiện ngã
"""
import os

# Đường dẫn
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
ALERTS_DIR = os.path.join(ROOT_DIR, 'alerts')

# Tạo thư mục nếu chưa tồn tại
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ALERTS_DIR, exist_ok=True)

# Cấu hình Camera
CAMERA_CONFIG = {
    'source': 0,  # 0 cho webcam mặc định, hoặc đường dẫn file video
    'width': 640,
    'height': 480,
    'fps': 30
}

# Cấu hình YOLOv8
YOLO_CONFIG = {
    'model': 'yolov8n.pt',  # yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
    'confidence': 0.5,
    'classes': [0],  # Chỉ phát hiện người (class 0 trong COCO dataset)
    'device': 'cpu'  # 'cpu' hoặc 'cuda'
}

# Cấu hình MediaPipe Pose
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 1,  # 0, 1, hoặc 2 (cao hơn = chính xác hơn nhưng chậm hơn)
    'smooth_landmarks': True,
    'enable_segmentation': False,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Cấu hình phát hiện ngã
FALL_DETECTION_CONFIG = {
    # Ngưỡng phát hiện
    'angle_threshold': 45,  # Góc nghiêng cơ thể (độ)
    'aspect_ratio_threshold': 1.2,  # Tỷ lệ width/height
    'velocity_threshold': 15,  # Vận tốc di chuyển ngang (pixels/frame)
    'vertical_velocity_threshold': 10,  # Vận tốc rơi dọc

    # Thời gian
    'fall_time_threshold': 1.0,  # Thời gian tối thiểu để xác nhận ngã (giây)
    'alert_cooldown': 10,  # Thời gian chờ giữa các cảnh báo (giây)

    # Các điểm khớp quan trọng (MediaPipe Pose landmarks)
    'key_points': {
        'nose': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28
    }
}

# Cấu hình cảnh báo
ALERT_CONFIG = {
    'sound_enabled': True,
    'visual_enabled': True,
    'save_fall_images': True,
    'log_falls': True,
    'alert_sound_file': os.path.join(ALERTS_DIR, 'alert.wav')
}

# Màu sắc cho hiển thị
COLORS = {
    'normal': (0, 255, 0),      # Xanh lá - Bình thường
    'warning': (0, 255, 255),    # Vàng - Cảnh báo
    'fall': (0, 0, 255),         # Đỏ - Ngã
    'text': (255, 255, 255),     # Trắng - Text
    'skeleton': (255, 200, 100)  # Cam - Khung xương
}