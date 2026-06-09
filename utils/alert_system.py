"""
Module hệ thống cảnh báo
"""
import cv2
import os
import time
from datetime import datetime
from config import ALERT_CONFIG, LOGS_DIR, ALERTS_DIR, COLORS


class AlertSystem:
    """
    Lớp quản lý cảnh báo khi phát hiện ngã
    """

    def __init__(self):
        self.config = ALERT_CONFIG
        self.log_file = os.path.join(LOGS_DIR, 'fall_events.log')

        # Tạo file log nếu chưa tồn tại
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("=== FALL DETECTION LOG ===\n")
                f.write("Created: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

    def play_alert_sound(self):
        """Phát âm thanh cảnh báo"""
        if not self.config['sound_enabled']:
            return

        try:
            # Thử phát âm thanh bằng Windows
            import winsound
            # Phát beep 3 lần
            for _ in range(3):
                winsound.Beep(1000, 500)  # 1000Hz, 500ms
        except ImportError:
            # Nếu không có winsound (Linux/Mac)
            print('\a' * 3)  # System beep
        except Exception as e:
            print(f"Could not play sound: {e}")

    def draw_alert(self, frame, fall_info):
        """
        Vẽ cảnh báo lên frame

        Args:
            frame: Frame ảnh
            fall_info: Thông tin về sự kiện ngã

        Returns:
            frame: Frame đã được vẽ cảnh báo
        """
        if not self.config['visual_enabled']:
            return frame

        h, w = frame.shape[:2]

        # Vẽ khung cảnh báo
        if fall_info['status'] == 'FALL':
            # Vẽ viền đỏ nhấp nháy
            cv2.rectangle(frame, (5, 5), (w-5, h-5), COLORS['fall'], 5)

            # Vẽ text cảnh báo lớn
            alert_text = "CANH BAO: PHAT HIEN NGA!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 3
            text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 60

            # Background cho text
            cv2.rectangle(frame,
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         COLORS['fall'], -1)

            cv2.putText(frame, alert_text, (text_x, text_y),
                       font, font_scale, COLORS['text'], thickness)

            # Thời gian ngã
            time_text = f"Thoi gian nga: {fall_info['time_falling']:.1f}s"
            cv2.putText(frame, time_text, (20, h - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['fall'], 2)

        elif fall_info['status'] == 'WARNING':
            # Vẽ viền vàng
            cv2.rectangle(frame, (5, 5), (w-5, h-5), COLORS['warning'], 3)

            # Text cảnh báo
            warning_text = "CANH BAO: Co the dang nga!"
            cv2.putText(frame, warning_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['warning'], 2)

        return frame

    def save_fall_image(self, frame, fall_info):
        """
        Lưu ảnh khi phát hiện ngã

        Args:
            frame: Frame ảnh
            fall_info: Thông tin về sự kiện ngã
        """
        if not self.config['save_fall_images']:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fall_{timestamp}.jpg"
        filepath = os.path.join(ALERTS_DIR, filename)

        cv2.imwrite(filepath, frame)
        print(f"Da luu anh: {filepath}")

    def log_fall_event(self, fall_info):
        """
        Ghi log sự kiện ngã

        Args:
            fall_info: Thông tin về sự kiện ngã
        """
        if not self.config['log_falls']:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = f"""
[{timestamp}] PHAT HIEN NGA
- Goc nghieng: {fall_info['body_angle']:.1f} do
- Ty le W/H: {fall_info['aspect_ratio']:.2f}
- Van toc: ({fall_info['velocity'][0]:.1f}, {fall_info['velocity'][1]:.1f})
- Thoi gian nga: {fall_info['time_falling']:.1f}s
- Diem chi so: {fall_info['fall_score']}
----------------------------------------
"""

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    def trigger_alert(self, frame, fall_info):
        """
        Kích hoạt tất cả các cảnh báo

        Args:
            frame: Frame ảnh
            fall_info: Thông tin về sự kiện ngã
        """
        if fall_info['status'] == 'FALL':
            self.play_alert_sound()
            self.save_fall_image(frame, fall_info)
            self.log_fall_event(fall_info)
            print(f"\n{'='*50}")
            print("CANH BAO: PHAT HIEN NGA NGUOI!")
            print(f"Goc nghieng: {fall_info['body_angle']:.1f} do")
            print(f"Ty le W/H: {fall_info['aspect_ratio']:.2f}")
            print(f"Thoi gian: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*50}\n")

    def draw_info_panel(self, frame, fall_info, fps):
        """
        Vẽ bảng thông tin lên frame

        Args:
            frame: Frame ảnh
            fall_info: Thông tin phát hiện
            fps: FPS hiện tại

        Returns:
            frame: Frame với bảng thông tin
        """
        h, w = frame.shape[:2]

        # Background panel (tăng chiều cao để chứa thêm thông tin cải tiến)
        panel_width = 280
        panel_height = 260  # Tăng từ 200 lên 260 để chứa thêm 2 dòng
        cv2.rectangle(frame, (w - panel_width - 10, 10),
                     (w - 10, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - panel_width - 10, 10),
                     (w - 10, panel_height), (255, 255, 255), 1)

        # Text thông tin
        y_offset = 35
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)

        # === CẬP NHẬT: Thêm thông tin cải tiến vào info panel ===
        # Lấy thông tin temporal voting và normalized velocity
        vote_count = fall_info.get('vote_count', 0)
        temporal_window = fall_info.get('temporal_window', 15)
        velocity_norm = fall_info.get('velocity_norm', (0, 0))

        info_lines = [
            f"FPS: {fps:.1f}",
            f"Trang thai: {fall_info['status']}",
            f"Goc nghieng: {fall_info['body_angle']:.1f} do",
            f"Ty le W/H: {fall_info['aspect_ratio']:.2f}",
            f"Van toc X: {fall_info['velocity'][0]:.1f} px/s",
            f"Van toc Y: {fall_info['velocity'][1]:.1f} px/s",
            # === Thông tin cải tiến ===
            f"VT chuan Y: {velocity_norm[1]:.2f}",  # Normalized velocity Y
            f"Vote: {vote_count}/{temporal_window}",  # Temporal voting
            f"Diem chi so: {fall_info['fall_score']}/4"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (w - panel_width, y_offset + i * line_height),
                       font, font_scale, color, 1)

        # Thanh trạng thái
        status_colors = {
            'NORMAL': COLORS['normal'],
            'WARNING': COLORS['warning'],
            'FALL': COLORS['fall']
        }

        status_color = status_colors.get(fall_info['status'], COLORS['normal'])
        cv2.rectangle(frame, (w - panel_width - 10, panel_height + 20),
                     (w - 10, panel_height + 40), status_color, -1)

        status_text = fall_info['status']
        text_size = cv2.getTextSize(status_text, font, 0.7, 2)[0]
        text_x = w - panel_width//2 - text_size[0]//2
        cv2.putText(frame, status_text, (text_x, panel_height + 35),
                   font, 0.7, (0, 0, 0), 2)

        return frame