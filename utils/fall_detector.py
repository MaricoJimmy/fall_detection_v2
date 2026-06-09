"""
Module phát hiện ngã sử dụng Pose Estimation
Đã cải tiến với:
- Chuẩn hóa vận tốc theo chiều cao bounding box
- Temporal voting (xét nhiều frame gần nhất)
- Post-fall confirmation (xác nhận người vẫn nằm sau ngã)
- Recovery logic (tự reset khi người đứng dậy)
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
    3. Vận tốc di chuyển (đã chuẩn hóa theo chiều cao cơ thể)
    4. Vị trí tương đối của các điểm khớp
    5. Temporal voting (xét nhiều frame gần nhất)
    6. Post-fall confirmation (xác nhận sau ngã)
    """

    def __init__(self):
        self.config = FALL_DETECTION_CONFIG

        # Lịch sử vị trí để tính vận tốc
        self.position_history = deque(maxlen=30)  # Lưu 30 frame gần nhất
        self.time_history = deque(maxlen=30)

        # === CẢI TIẾN: Temporal voting ===
        # Lưu kết quả "có nghi ngờ ngã không" của N frame gần nhất
        # Dùng để tránh báo nhầm do 1 frame bị lỗi pose
        self.vote_history = deque(maxlen=self.config['temporal_window_size'])

        # Trạng thái
        self.fall_start_time = None
        self.is_falling = False
        self.fall_confirmed = False
        self.last_alert_time = 0

        # === CẢI TIẾN: Post-fall confirmation ===
        # Theo dõi thời gian người vẫn nằm sau khi ngã
        self.post_fall_start_time = None

        # === CẢI TIẾN: Recovery logic ===
        # Đếm số frame liên tiếp người đứng dậy (góc < recovery_angle)
        self.recovery_count = 0

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

    def detect(self, landmarks, bbox, frame_width, frame_height, timestamp=None):
        """
        Phát hiện ngã chính (đã cải tiến)

        Args:
            landmarks: MediaPipe pose landmarks
            bbox: [x1, y1, x2, y2] normalized bounding box (0-1)
            frame_width: Chiều rộng frame
            frame_height: Chiều cao frame
            timestamp: Timestamp của frame (giây). Nếu None, dùng time.time()

        Returns:
            dict: Kết quả phát hiện
        """
        current_time = timestamp if timestamp is not None else time.time()

        # Chuyển đổi bbox sang pixel
        bbox_pixel = [
            bbox[0] * frame_width,
            bbox[1] * frame_height,
            bbox[2] * frame_width,
            bbox[3] * frame_height
        ]

        # === CẢI TIẾN 1: Tính chiều cao bounding box để chuẩn hóa vận tốc ===
        # Chiều cao bbox đại diện cho chiều cao cơ thể trong ảnh
        # Dùng để chuẩn hóa vận tốc, giúp không phụ thuộc khoảng cách camera
        bbox_height = bbox_pixel[3] - bbox_pixel[1]

        # Tính toán các chỉ số
        body_angle = self.calculate_body_angle(landmarks)
        aspect_ratio = self.calculate_aspect_ratio(bbox_pixel)

        # Tính toán vận tốc (pixel/giây)
        center_x = (bbox_pixel[0] + bbox_pixel[2]) / 2
        center_y = (bbox_pixel[1] + bbox_pixel[3]) / 2
        vx, vy = self.calculate_velocity([center_x, center_y], current_time)

        # === CẢI TIẾN 1: Chuẩn hóa vận tốc theo chiều cao cơ thể ===
        # vy_norm = 0.5 nghĩa là di chuyển xuống bằng nửa chiều cao cơ thể mỗi giây
        # Giúp hệ thống hoạt động ổn định dù người đứng gần hay xa camera
        vx_norm = vx / (bbox_height + 1e-6)
        vy_norm = vy / (bbox_height + 1e-6)

        # Lưu lịch sử
        self.position_history.append([center_x, center_y])
        self.time_history.append(current_time)

        # Kiểm tra ngã bằng điểm khớp
        keypoint_fall = self.check_fall_by_keypoints(landmarks)

        # Xác định các chỉ số cho thấy đang ngã
        # Lưu ý: high_velocity giờ dùng normalized velocity thay vì pixel velocity
        fall_indicators = {
            'angle_exceeded': body_angle > self.angle_threshold,
            'aspect_ratio_exceeded': aspect_ratio > self.aspect_ratio_threshold,
            'high_velocity': abs(vy_norm) > self.config['normalized_velocity_threshold'],
            'keypoint_fall': keypoint_fall
        }

        # Đếm số chỉ số cho thấy đang ngã
        fall_score = sum(fall_indicators.values())

        # === CẢI TIẾN 2: Temporal voting ===
        # Thay vì chỉ nhìn 1 frame, hệ thống nhìn N frame gần nhất
        # Lưu kết quả "có >= 2 chỉ số nghi ngờ ngã không" vào lịch sử
        is_single_frame_suspicious = fall_score >= 2
        self.vote_history.append(is_single_frame_suspicious)

        # Đếm số frame nghi ngờ trong cửa sổ temporal
        vote_count = sum(self.vote_history)

        # Chỉ coi là "nghi ngờ ngã" khi có đủ số frame nghi ngờ
        # Ví dụ: cần 10/15 frame gần nhất nghi ngờ mới bắt đầu đếm
        is_potential_fall = vote_count >= self.config['temporal_vote_threshold']

        # === CẢI TIẾN 4: Recovery logic - kiểm tra người đang đứng dậy ===
        # Nếu góc nghiêng nhỏ (< recovery_angle), người có thể đang đứng dậy
        recovery_angle = self.config['recovery_angle']
        recovery_frames = self.config['recovery_frames']

        if body_angle < recovery_angle:
            # Góc nhỏ = người đang thẳng đứng, đếm frame recovery
            self.recovery_count += 1
        else:
            # Góc lớn = người vẫn nghiêng, reset recovery count
            self.recovery_count = 0

        # Nếu đứng dậy liên tục đủ lâu => reset hoàn toàn về NORMAL
        if self.recovery_count >= recovery_frames:
            self.is_falling = False
            self.fall_start_time = None
            self.fall_confirmed = False
            self.post_fall_start_time = None
            self.recovery_count = 0
            self.vote_history.clear()

        # Logic phát hiện ngã (với temporal voting)
        if is_potential_fall and not self.is_falling:
            self.is_falling = True
            self.fall_start_time = current_time

        if self.is_falling:
            time_falling = current_time - self.fall_start_time

            # Xác nhận ngã khi đủ thời gian VÀ vẫn còn nghi ngờ
            if is_potential_fall and time_falling >= self.fall_time_threshold:
                self.fall_confirmed = True

                # === CẢI TIẾN 3: Post-fall confirmation ===
                # Bắt đầu đếm thời gian post-fall khi lần đầu xác nhận ngã
                if self.post_fall_start_time is None:
                    self.post_fall_start_time = current_time

            elif not is_potential_fall:
                # Không còn đủ frame nghi ngờ => reset
                self.is_falling = False
                self.fall_start_time = None
                self.fall_confirmed = False
                self.post_fall_start_time = None

        # === CẢI TIẾN 3: Kiểm tra post-fall ===
        # Sau khi xác nhận ngã, kiểm tra người có vẫn nằm/nghiêng không
        post_fall_active = False
        if self.fall_confirmed and self.post_fall_start_time is not None:
            post_fall_duration = current_time - self.post_fall_start_time

            # Nếu người vẫn nghiêng (góc lớn) => vẫn trong trạng thái post-fall
            if body_angle > self.angle_threshold:
                post_fall_active = True
                # Nếu đã nằm đủ lâu => giữ trạng thái FALL
                if post_fall_duration >= self.config['post_fall_time']:
                    pass  # Giữ fall_confirmed = True
            else:
                # Người đã đứng dậy => có thể không phải ngã thật
                # Reset nếu chưa đủ post_fall_time
                if post_fall_duration < self.config['post_fall_time']:
                    self.fall_confirmed = False
                    self.post_fall_start_time = None

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
            'velocity': (vx, vy),  # Vận tốc gốc (pixel/giây)
            'velocity_norm': (vx_norm, vy_norm),  # Vận tốc chuẩn hóa
            'fall_indicators': fall_indicators,
            'fall_score': fall_score,
            'time_falling': time_falling if self.is_falling else 0,
            # === Thông tin cải tiến để hiển thị ===
            'vote_count': vote_count,  # Số frame nghi ngờ trong cửa sổ
            'temporal_window': self.config['temporal_window_size'],  # Kích thước cửa sổ
            'post_fall_active': post_fall_active,  # Đang trong giai đoạn post-fall
            'recovery_count': self.recovery_count  # Số frame đứng dậy liên tiếp
        }

    def reset(self):
        """Reset trạng thái detector"""
        self.position_history.clear()
        self.time_history.clear()
        self.vote_history.clear()  # Clear cả vote history
        self.fall_start_time = None
        self.is_falling = False
        self.fall_confirmed = False
        self.post_fall_start_time = None  # Reset post-fall
        self.recovery_count = 0  # Reset recovery

    def can_alert(self, timestamp=None):
        """
        Kiểm tra xem có thể gửi cảnh báo không (tránh spam)

        Returns:
            bool: True nếu có thể cảnh báo
        """
        current_time = timestamp if timestamp is not None else time.time()
        if current_time - self.last_alert_time >= self.config['alert_cooldown']:
            self.last_alert_time = current_time
            return True
        return False
