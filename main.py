"""
AI-Based Human Fall Detection System Using Computer Vision
Sử dụng YOLOv8, MediaPipe và Pose Estimation

File chính để chạy demo hệ thống phát hiện ngã
"""
import cv2
import numpy as np
import time
from datetime import datetime
from config import (
    CAMERA_CONFIG, COLORS,
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
        print("  Su dung: YOLOv8-pose + Pose Estimation (rule-based)")
        print("=" * 60)
        print()

        # Khởi tạo Pose Estimator (YOLOv8-pose)
        print("[1/2] Dang tai YOLOv8-pose model...")
        self.pose_estimator = PoseEstimator()
        print("      -> YOLOv8-pose san sang!")

        # Khởi tạo Fall Detector
        print("[2/2] Dang khoi tao Fall Detector...")
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

    def process_frame(self, frame, timestamp=None):
        """
        Xử lý một frame

        Args:
            frame: Frame ảnh BGR

        Returns:
            tuple: (processed_frame, fall_info)
        """
        h, w = frame.shape[:2]

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

        # Pose estimation + detection (YOLOv8-pose)
        landmarks, bbox_norm, bbox_pixel = self.pose_estimator.process_frame(frame)

        if landmarks and bbox_norm:
            # Vẽ landmarks
            frame = self.pose_estimator.draw_landmarks(frame, landmarks)

            # Vẽ bounding box
            if bbox_pixel:
                cv2.rectangle(frame, (bbox_pixel[0], bbox_pixel[1]),
                             (bbox_pixel[2], bbox_pixel[3]), (0, 255, 0), 2)

            # Phát hiện ngã
            fw = bbox_pixel[2] - bbox_pixel[0] if bbox_pixel else w
            fh = bbox_pixel[3] - bbox_pixel[1] if bbox_pixel else h
            fall_info = self.fall_detector.detect(
                landmarks, bbox_norm, fw, fh,
                timestamp=timestamp
            )

            # Vẽ cảnh báo
            frame = self.alert_system.draw_alert(frame, fall_info)

            # Kích hoạt cảnh báo nếu ngã
            if fall_info['status'] == 'FALL' and self.fall_detector.can_alert(timestamp=timestamp):
                self.alert_system.trigger_alert(frame.copy(), fall_info)

        return frame, fall_info

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
    args = parser.parse_args()

    system = FallDetectionSystem()
    system.run(source=args.source)


if __name__ == "__main__":
    main()