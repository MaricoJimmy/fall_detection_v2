"""
AI-Based Human Fall Detection System Using Computer Vision
Sử dụng YOLOv8, MediaPipe và Pose Estimation

File chính để chạy demo hệ thống phát hiện ngã
"""
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO

from config import (
    CAMERA_CONFIG, YOLO_CONFIG, COLORS,
    LOGS_DIR
)
from utils import FallDetector, PoseEstimator, AlertSystem


class FallDetectionSystem:
    """
    Hệ thống phát hiện ngã chính
    """

    def __init__(self):
        """Khởi tạo hệ thống"""
        print("=" * 60)
        print("  AI-BASED HUMAN FALL DETECTION SYSTEM")
        print("  Su dung: YOLOv8 + MediaPipe + Pose Estimation")
        print("=" * 60)
        print()

        # Khởi tạo YOLOv8
        print("[1/3] Dang tai YOLOv8 model...")
        self.yolo_model = YOLO(YOLO_CONFIG['model'])
        print(f"      -> Da tai {YOLO_CONFIG['model']} thanh cong!")

        # Khởi tạo Pose Estimator
        print("[2/3] Dang khoi tao MediaPipe Pose...")
        self.pose_estimator = PoseEstimator()
        print("      -> MediaPipe Pose san sang!")

        # Khởi tạo Fall Detector
        print("[3/3] Dang khoi tao Fall Detector...")
        self.fall_detector = FallDetector()
        print("      -> Fall Detector san sang!")

        # Khởi tạo Alert System
        self.alert_system = AlertSystem()

        # Biến theo dõi
        self.cap = None
        self.running = False
        self.fps_history = []
        self.frame_count = 0

        print()
        print("He thong san sang!")
        print("-" * 60)

    def init_camera(self):
        """Khởi tạo camera"""
        source = CAMERA_CONFIG['source']
        
        # Thử ép kiểu sang int nếu là camera index dạng chuỗi (ví dụ "0")
        try:
            source = int(source)
            CAMERA_CONFIG['source'] = source
        except ValueError:
            pass
            
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Khong the mo camera/video: {source}"
            )

        # Chỉ thiết lập độ phân giải nếu là webcam (số nguyên)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
            print(f"Camera khoi tao: {CAMERA_CONFIG['width']}x{CAMERA_CONFIG['height']}")
        else:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video file khoi tao: {w}x{h}")

    def detect_person_yolo(self, frame):
        """
        Phát hiện người sử dụng YOLOv8

        Args:
            frame: Frame ảnh BGR

        Returns:
            list: Danh sách các bounding box [x1, y1, x2, y2, confidence]
        """
        results = self.yolo_model(
            frame,
            conf=YOLO_CONFIG['confidence'],
            classes=YOLO_CONFIG['classes'],
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])

        return detections

    def draw_yolo_detection(self, frame, detections):
        """
        Vẽ bounding box YOLO lên frame

        Args:
            frame: Frame ảnh
            detections: Danh sách các detection

        Returns:
            frame: Frame đã vẽ
        """
        for det in detections:
            x1, y1, x2, y2, conf = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ label
            label = f"Person {conf:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        return frame

    def process_frame(self, frame, timestamp=None):
        """
        Xử lý một frame

        Args:
            frame: Frame ảnh BGR

        Returns:
            tuple: (processed_frame, fall_info)
        """
        h, w = frame.shape[:2]

        # Bước 1: Phát hiện người bằng YOLO
        detections = self.detect_person_yolo(frame)

        # Default fall_info
        fall_info = {
            'status': 'NORMAL',
            'color': COLORS['normal'],
            'is_falling': False,
            'fall_confirmed': False,
            'body_angle': 0,
            'aspect_ratio': 1.0,
            'velocity': (0, 0),
            'fall_indicators': {},
            'fall_score': 0,
            'time_falling': 0
        }

        # Vẽ YOLO detections
        frame = self.draw_yolo_detection(frame, detections)

        # Bước 2: Nếu có người, thực hiện pose estimation
        if detections:
            # Lấy detection đầu tiên (giả sử 1 người)
            det = detections[0]
            x1, y1, x2, y2 = det[:4]

            # Crop region chứa người (mở rộng một chút)
            padding = 20
            x1_crop = max(0, int(x1) - padding)
            y1_crop = max(0, int(y1) - padding)
            x2_crop = min(w, int(x2) + padding)
            y2_crop = min(h, int(y2) + padding)

            person_region = frame[y1_crop:y2_crop, x1_crop:x2_crop]

            if person_region.size > 0:
                # Pose estimation
                landmarks, bbox_norm, bbox_pixel = self.pose_estimator.process_frame(
                    person_region
                )

                if landmarks:
                    # Vẽ landmarks lên frame gốc
                    # Cần điều chỉnh tọa độ
                    self.draw_landmarks_on_frame(
                        frame, landmarks, x1_crop, y1_crop,
                        x2_crop - x1_crop, y2_crop - y1_crop
                    )

                    # Phát hiện ngã
                    fall_info = self.fall_detector.detect(
                        landmarks, bbox_norm,
                        x2_crop - x1_crop, y2_crop - y1_crop,
                        timestamp=timestamp
                    )

                    # Vẽ cảnh báo
                    frame = self.alert_system.draw_alert(frame, fall_info)

                    # Kích hoạt cảnh báo nếu ngã
                    if fall_info['status'] == 'FALL' and self.fall_detector.can_alert(timestamp=timestamp):
                        self.alert_system.trigger_alert(frame.copy(), fall_info)

        return frame, fall_info

    def draw_landmarks_on_frame(self, frame, landmarks, offset_x, offset_y,
                                region_w, region_h):
        """
        Vẽ landmarks lên frame gốc (với offset)

        Args:
            frame: Frame gốc
            landmarks: MediaPipe landmarks
            offset_x, offset_y: Offset từ crop region
            region_w, region_h: Kích thước crop region
        """
        for idx, landmark in enumerate(landmarks):
            if landmark.visibility > 0.5:
                x = int(landmark.x * region_w + offset_x)
                y = int(landmark.y * region_h + offset_y)
                cv2.circle(frame, (x, y), 3, COLORS['skeleton'], -1)

        # Vẽ các đường nối chính
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (23, 25), (24, 26), (25, 27), (26, 28),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 11), (0, 12)
        ]

        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            if start.visibility > 0.5 and end.visibility > 0.5:
                start_point = (
                    int(start.x * region_w + offset_x),
                    int(start.y * region_h + offset_y)
                )
                end_point = (
                    int(end.x * region_w + offset_x),
                    int(end.y * region_h + offset_y)
                )
                cv2.line(frame, start_point, end_point, COLORS['skeleton'], 2)

    def calculate_fps(self):
        """Tính FPS"""
        if len(self.fps_history) < 2:
            return 0

        fps = len(self.fps_history) / (self.fps_history[-1] - self.fps_history[0])
        return fps

    def run(self, source=None):
        """
        Chạy hệ thống phát hiện ngã

        Args:
            source: Nguồn video (None = webcam, hoặc đường dẫn file video)
        """
        if source is not None:
            CAMERA_CONFIG['source'] = source

        self.init_camera()
        self.running = True

        print("\n" + "=" * 60)
        print("  BAT DAU CHAY HE THONG")
        print("=" * 60)
        print()
        print("Phim tat:")
        print("  [q] - Thoat")
        print("  [r] - Reset detector")
        print("  [s] - Luu anh")
        print("  [SPACE] - Tam dung/Tiep tuc")
        print()
        print("-" * 60)

        paused = False
        last_alert_time = 0

        try:
            while self.running:
                if not paused:
                    ret, frame = self.cap.read()

                    if not ret:
                        print("Khong the doc frame tu camera!")
                        break

                    # Ghi nhận thời gian
                    self.fps_history.append(time.time())
                    if len(self.fps_history) > 30:
                        self.fps_history.pop(0)

                    # Xử lý frame
                    is_video_file = isinstance(CAMERA_CONFIG['source'], str)
                    timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_video_file else time.time()
                    processed_frame, fall_info = self.process_frame(frame, timestamp=timestamp)

                    # Tính FPS
                    fps = self.calculate_fps()

                    # Vẽ bảng thông tin
                    processed_frame = self.alert_system.draw_info_panel(
                        processed_frame, fall_info, fps
                    )

                    # Vẽ thời gian
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(
                        processed_frame, timestamp, (20, processed_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                    )

                    # Hiển thị
                    cv2.imshow("Fall Detection System", processed_frame)

                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nDang thoat...")
                    break
                elif key == ord('r'):
                    self.fall_detector.reset()
                    print("Detector da reset!")
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(LOGS_DIR, f"screenshot_{timestamp}.jpg")
                    cv2.imwrite(filename, processed_frame)
                    print(f"Da luu anh: {filename}")
                elif key == ord(' '):
                    paused = not paused
                    print("Tam pause" if paused else "Tiep tuc")

        except KeyboardInterrupt:
            print("\n\nNhan Ctrl+C, dang thoat...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Dọn dẹp tài nguyên"""
        print("Dang don dep tai nguyen...")

        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()
        self.pose_estimator.close()

        print("Da don dep xong. Tam biet!")


def main():
    """Hàm main"""
    import argparse

    parser = argparse.ArgumentParser(
        description='AI-Based Human Fall Detection System'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        default=None,
        help='Nguon video (duong dan file hoac camera index). Mac dinh: webcam (0)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolov8n.pt',
        help='YOLOv8 model (yolov8n.pt, yolov8s.pt, yolov8m.pt). Mac dinh: yolov8n.pt'
    )

    args = parser.parse_args()

    # Cập nhật config
    if args.model:
        YOLO_CONFIG['model'] = args.model

    # Chạy hệ thống
    system = FallDetectionSystem()
    system.run(source=args.source)


if __name__ == "__main__":
    main()