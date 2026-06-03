"""
Module phát hiện ngã sử dụng Pose Estimation
"""
import numpy as np
import time
from collections import deque
from config import FALL_DETECTION_CONFIG, COLORS


class FallDetector:
    """
    Lớp phát hiện ngã dựa trên Pose Estimation

    Sử dụng nhiều tiêu chí:
    1. Góc nghiêng của cơ thể
    2. Tỷ lệ width/height của bounding box
    3. Vận tốc di chuyển
    4. Vị trí tương đối của các điểm khớp
    """

    def __init__(self):
        self.config = FALL_DETECTION_CONFIG

        # Lịch sử vị trí để tính vận tốc
        self.position_history = deque(maxlen=30)  # Lưu 30 frame gần nhất
        self.time_history = deque(maxlen=30)

        # Trạng thái
        self.fall_start_time = None
        self.is_falling = False
        self.fall_confirmed = False
        self.last_alert_time = 0

        # Các ngưỡng
        self.angle_threshold = self.config['angle_threshold']
        self.aspect_ratio_threshold = self.config['aspect_ratio_threshold']
        self.velocity_threshold = self.config['velocity_threshold']
        self.fall_time_threshold = self.config['fall_time_threshold']

    def calculate_body_angle(self, landmarks):
        """
        Tính góc nghiêng của cơ thể so với phương thẳng đứng

        Args:
            landmarks: MediaPipe pose landmarks

        Returns:
            float: Góc nghiêng (độ)
        """
        # Lấy các điểm khớp quan trọng
        kp = self.config['key_points']

        # Tính trung tâm vai và hông
        left_shoulder = landmarks[kp['left_shoulder']]
        right_shoulder = landmarks[kp['right_shoulder']]
        left_hip = landmarks[kp['left_hip']]
        right_hip = landmarks[kp['right_hip']]

        # Tính trung điểm
        shoulder_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])

        hip_center = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ])

        # Tính vector đường trục cơ thể
        body_vector = shoulder_center - hip_center

        # Vector thẳng đứng đi lên (0, -1)
        vertical_vector = np.array([0, -1])

        # Tính góc giữa hai vector
        cos_angle = np.dot(body_vector, vertical_vector) / (
            np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector) + 1e-6
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def calculate_aspect_ratio(self, bbox):
        """
        Tính tỷ lệ width/height của bounding box

        Khi người đứng: height > width -> aspect_ratio < 1
        Khi người nằm ngang: width > height -> aspect_ratio > 1

        Args:
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            float: Tỷ lệ width/height
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if height < 1e-6:
            return 1.0

        return width / height

    def calculate_velocity(self, current_pos, current_time):
        """
        Tính vận tốc di chuyển của người

        Args:
            current_pos: Vị trí trung tâm hiện tại [x, y]
            current_time: Thời điểm hiện tại

        Returns:
            tuple: (velocity_horizontal, velocity_vertical)
        """
        if len(self.position_history) == 0:
            return 0, 0

        prev_pos = self.position_history[-1]
        prev_time = self.time_history[-1]

        dt = current_time - prev_time
        if dt < 1e-6:
            return 0, 0

        # Vận tốc ngang (dương = sang phải, âm = sang trái)
        vx = (current_pos[0] - prev_pos[0]) / dt

        # Vận tốc dọc (dương = đi xuống, âm = đi lên)
        vy = (current_pos[1] - prev_pos[1]) / dt

        return vx, vy

    def check_fall_by_keypoints(self, landmarks):
        """
        Kiểm tra ngã dựa trên vị trí các điểm khớp

        Args:
            landmarks: MediaPipe pose landmarks

        Returns:
            bool: True nếu có khả năng ngã
        """
        kp = self.config['key_points']

        # Lấy các điểm khớp
        nose = landmarks[kp['nose']]
        left_ankle = landmarks[kp['left_ankle']]
        right_ankle = landmarks[kp['right_ankle']]
        left_hip = landmarks[kp['left_hip']]
        right_hip = landmarks[kp['right_hip']]

        # Kiểm tra: mắt cá chân cao hơn hông (người đang ngã hoặc đã ngã)
        hip_center_y = (left_hip.y + right_hip.y) / 2
        ankle_center_y = (left_ankle.y + right_ankle.y) / 2

        # Trong hệ tọa độ ảnh, y tăng từ trên xuống dưới
        # Nếu ankle_y < hip_y nghĩa là mắt cá chân cao hơn hông
        ankle_above_hip = ankle_center_y < hip_center_y

        # Kiểm tra độ sâu z của các điểm (nếu có)
        # Khi ngã, các điểm có độ sâu khác nhau rõ rệt
        if hasattr(left_ankle, 'z') and hasattr(right_ankle, 'z'):
            depth_variation = abs(left_ankle.z - right_ankle.z)
            unusual_depth = depth_variation > 0.3
        else:
            unusual_depth = False

        return ankle_above_hip or unusual_depth

    def detect(self, landmarks, bbox, frame_width, frame_height):
        """
        Phát hiện ngã chính

        Args:
            landmarks: MediaPipe pose landmarks
            bbox: [x1, y1, x2, y2] normalized bounding box (0-1)
            frame_width: Chiều rộng frame
            frame_height: Chiều cao frame

        Returns:
            dict: Kết quả phát hiện
        """
        current_time = time.time()

        # Chuyển đổi bbox sang pixel
        bbox_pixel = [
            bbox[0] * frame_width,
            bbox[1] * frame_height,
            bbox[2] * frame_width,
            bbox[3] * frame_height
        ]

        # Tính toán các chỉ số
        body_angle = self.calculate_body_angle(landmarks)
        aspect_ratio = self.calculate_aspect_ratio(bbox_pixel)

        # Tính toán vận tốc
        center_x = (bbox_pixel[0] + bbox_pixel[2]) / 2
        center_y = (bbox_pixel[1] + bbox_pixel[3]) / 2
        vx, vy = self.calculate_velocity([center_x, center_y], current_time)

        # Lưu lịch sử
        self.position_history.append([center_x, center_y])
        self.time_history.append(current_time)

        # Kiểm tra ngã bằng điểm khớp
        keypoint_fall = self.check_fall_by_keypoints(landmarks)

        # Xác định trạng thái
        fall_indicators = {
            'angle_exceeded': body_angle > self.angle_threshold,
            'aspect_ratio_exceeded': aspect_ratio > self.aspect_ratio_threshold,
            'high_velocity': abs(vy) > self.config['vertical_velocity_threshold'],
            'keypoint_fall': keypoint_fall
        }

        # Đếm số chỉ số cho thấy đang ngã
        fall_score = sum(fall_indicators.values())

        # Logic phát hiện ngã
        is_potential_fall = fall_score >= 2  # Ít nhất 2 chỉ số cho thấy ngã

        if is_potential_fall and not self.is_falling:
            self.is_falling = True
            self.fall_start_time = current_time

        if self.is_falling:
            time_falling = current_time - self.fall_start_time

            if is_potential_fall and time_falling >= self.fall_time_threshold:
                self.fall_confirmed = True
            elif not is_potential_fall:
                # Reset nếu không còn dấu hiệu ngã
                self.is_falling = False
                self.fall_start_time = None
                self.fall_confirmed = False

        # Xác định trạng thái cuối cùng
        if self.fall_confirmed:
            status = 'FALL'
            color = COLORS['fall']
        elif self.is_falling:
            status = 'WARNING'
            color = COLORS['warning']
        else:
            status = 'NORMAL'
            color = COLORS['normal']

        return {
            'status': status,
            'color': color,
            'is_falling': self.is_falling,
            'fall_confirmed': self.fall_confirmed,
            'body_angle': body_angle,
            'aspect_ratio': aspect_ratio,
            'velocity': (vx, vy),
            'fall_indicators': fall_indicators,
            'fall_score': fall_score,
            'time_falling': time_falling if self.is_falling else 0
        }

    def reset(self):
        """Reset trạng thái detector"""
        self.position_history.clear()
        self.time_history.clear()
        self.fall_start_time = None
        self.is_falling = False
        self.fall_confirmed = False

    def can_alert(self):
        """
        Kiểm tra xem có thể gửi cảnh báo không (tránh spam)

        Returns:
            bool: True nếu có thể cảnh báo
        """
        current_time = time.time()
        if current_time - self.last_alert_time >= self.config['alert_cooldown']:
            self.last_alert_time = current_time
            return True
        return False