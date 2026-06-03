"""
Module Pose Estimation sử dụng MediaPipe
"""
import cv2
import mediapipe as mp
from config import MEDIAPIPE_CONFIG


class PoseEstimator:
    """
    Lớp thực hiện Pose Estimation sử dụng MediaPipe
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Khởi tạo MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=MEDIAPIPE_CONFIG['static_image_mode'],
            model_complexity=MEDIAPIPE_CONFIG['model_complexity'],
            smooth_landmarks=MEDIAPIPE_CONFIG['smooth_landmarks'],
            enable_segmentation=MEDIAPIPE_CONFIG['enable_segmentation'],
            min_detection_confidence=MEDIAPIPE_CONFIG['min_detection_confidence'],
            min_tracking_confidence=MEDIAPIPE_CONFIG['min_tracking_confidence']
        )

    def process_frame(self, frame):
        """
        Xử lý frame để detect pose

        Args:
            frame: Frame ảnh BGR từ OpenCV

        Returns:
            tuple: (landmarks, bbox_normalized, bbox_pixel)
        """
        # Chuyển BGR sang RGB cho MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.pose.process(frame_rgb)

        landmarks = None
        bbox_normalized = None
        bbox_pixel = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Tính bounding box từ các landmarks
            h, w, _ = frame.shape

            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]

            # Bounding box với padding
            padding = 0.05
            x_min = max(0, min(x_coords) - padding)
            x_max = min(1, max(x_coords) + padding)
            y_min = max(0, min(y_coords) - padding)
            y_max = min(1, max(y_coords) + padding)

            bbox_normalized = [x_min, y_min, x_max, y_max]
            bbox_pixel = [
                int(x_min * w),
                int(y_min * h),
                int(x_max * w),
                int(y_max * h)
            ]

        return landmarks, bbox_normalized, bbox_pixel

    def draw_landmarks(self, frame, landmarks):
        """
        Vẽ các điểm khớp và khung xương lên frame

        Args:
            frame: Frame ảnh
            landmarks: MediaPipe pose landmarks

        Returns:
            frame: Frame đã được vẽ
        """
        if landmarks is None:
            return frame

        # Tạo pose landmarks object cho drawing
        h, w, _ = frame.shape

        # Vẽ các điểm khớp
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            # Chỉ vẽ các điểm có độ tin cậy cao
            if landmark.visibility > 0.5:
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # Vẽ các đường nối
        connections = [
            (11, 12),  # Vai trái - Vai phải
            (11, 23),  #<think>Vai trái - Hông trái
            (12, 24),  # Vai phải - Hông phải
            (23, 24),  # Hông trái - Hông phải
            (23, 25),  # Hông trái - Gối trái
            (24, 26),  # Hông phải - Gối phải
            (25, 27),  # Gối trái - Cổ chân trái
            (26, 28),  # Gối phải - Cổ chân phải
            (11, 13),  # Vai trái - Khủy tay trái
            (13, 15),  # Khủy tay trái - Cổ tay trái
            (12, 14),  # Vai phải - Khủy tay phải
            (14, 16),  # Khurdy tay phải - Cổ tay phải
            (0, 11),   # Mũi - Vai trái
            (0, 12),   # Mũi - Vai phải
        ]

        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            if start.visibility > 0.5 and end.visibility > 0.5:
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (255, 200, 100), 2)

        return frame

    def close(self):
        """Đóng pose estimator"""
        self.pose.close()