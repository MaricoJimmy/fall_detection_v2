"""
AI-Based Human Fall Detection System v2
Sử dụng YOLOv8-Pose + FallTCN (Velocity Features)
Hỗ trợ Webcam Real-time và Video file

Usage:
    python main_v2.py                    # Chạy webcam mặc định
    python main_v2.py --source 0         # Chạy webcam index 0
    python main_v2.py --source video.mp4 # Chạy từ file video
    python main_v2.py --threshold 0.6    # Tùy chỉnh ngưỡng phát hiện
"""
import os
import sys
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
import time
from datetime import datetime

# Import model TCN + Velocity
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_tcn_veloci import FallTCN, add_velocity_features

# ============================================================
#  CẤU HÌNH HỆ THỐNG
# ============================================================
CONFIG = {
    'model_path': 'models/skeleton_veloci/TCN_Veloci_best.pth',
    'yolo_model': 'yolov8m-pose.pt',
    'window_size': 30,
    'num_features': 102,   # 51 spatial + 51 velocity
    'threshold': 0.5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'camera_width': 640,
    'camera_height': 480,
    'camera_fps': 30,
}

# Keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
MIN_CONF = 0.3

# Skeleton connections cho vẽ khung xương
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
    (5, 11), (6, 12), (11, 12),              # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)   # Legs
]

# Màu sắc giao diện
COLORS = {
    'normal': (0, 255, 0),       # Xanh lá
    'fall': (0, 0, 255),         # Đỏ
    'skeleton': (255, 200, 100), # Vàng nhạt
    'text_bg': (0, 0, 0),       # Đen
    'text_white': (255, 255, 255),
    'panel_bg': (40, 40, 40),    # Xám đậm
}


class FallDetectionSystemV2:
    """
    Hệ thống phát hiện Té ngã Thời gian thực v2
    Sử dụng YOLOv8-Pose + FallTCN (Velocity Features)
    Hỗ trợ cả Webcam và Video file
    """

    def __init__(self):
        print("=" * 65)
        print("  FALL DETECTION SYSTEM v2")
        print("  YOLOv8-Pose + FallTCN (Velocity Features)")
        print("=" * 65)
        print(f"  Device: {CONFIG['device']}")
        if CONFIG['device'] == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print()

        # ---- Load FallTCN model ----
        print("[1/2] Loading FallTCN (Velocity) model...")
        self.model = FallTCN(
            input_size=CONFIG['num_features'],
            num_channels=[64, 128, 128],
            kernel_size=3,
            dropout=0.3
        )
        checkpoint = torch.load(
            CONFIG['model_path'],
            map_location=CONFIG['device'],
            weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(CONFIG['device'])
        self.model.eval()
        print("      -> FallTCN loaded successfully!")

        # ---- Load YOLOv8-Pose ----
        print("[2/2] Loading YOLOv8-Pose model...")
        self.yolo = YOLO(CONFIG['yolo_model'])
        print("      -> YOLOv8-Pose loaded successfully!")

        # ---- Tracking & Buffer cho từng người ----
        self.skeleton_buffers = {}     # {person_id: deque of skeletons}
        self.fall_probabilities = {}   # {person_id: float}
        self.statuses = {}             # {person_id: "NORMAL" | "FALL"}
        self.tracked_bboxes = {}       # {person_id: [x1,y1,x2,y2]}
        self.fall_alert_timers = {}    # {person_id: int (countdown frames)}
        self.next_id = 0
        self.iou_threshold = 0.3
        self.fall_alert_duration = 60  # Giữ cảnh báo FALL trong 60 frames (~2s)

        # ---- FPS tracking ----
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0

        print()
        print("=" * 65)
        print("  HE THONG SAN SANG!")
        print("=" * 65)

    # ============================================================
    #  POSE EXTRACTION & NORMALIZATION
    # ============================================================
    def detect_persons(self, frame):
        """Phát hiện người và keypoints bằng YOLOv8-Pose"""
        results = self.yolo(frame, conf=0.3, verbose=False)

        detections = []
        keypoints_dict = {}

        for result in results:
            if result.boxes is None:
                continue
            kps_xy = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
            kps_conf = result.keypoints.conf.cpu().numpy() if result.keypoints is not None else []

            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])

                skeleton = np.zeros((17, 3), dtype=np.float32)
                if i < len(kps_xy):
                    skeleton[:, :2] = kps_xy[i]
                    skeleton[:, 2] = kps_conf[i]
                keypoints_dict[len(detections) - 1] = skeleton

        return detections, keypoints_dict

    def normalize_skeleton(self, skeleton):
        """Chuẩn hóa skeleton về tâm hông, chia cho chiều dài thân"""
        if skeleton is None:
            return None

        skeleton = skeleton.copy()
        xy = skeleton[:, :2]
        conf = skeleton[:, 2]

        # Tìm tâm hông (Hip Center)
        lh = xy[LEFT_HIP]; rh = xy[RIGHT_HIP]
        lhc = conf[LEFT_HIP]; rhc = conf[RIGHT_HIP]

        if lhc < MIN_CONF and rhc < MIN_CONF:
            return None

        if lhc >= MIN_CONF and rhc >= MIN_CONF:
            hip_center = (lh + rh) / 2
        elif lhc >= MIN_CONF:
            hip_center = lh
        else:
            hip_center = rh

        # Tìm tâm vai (Shoulder Center) để tính Body Scale
        ls = xy[LEFT_SHOULDER]; rs = xy[RIGHT_SHOULDER]
        lsc = conf[LEFT_SHOULDER]; rsc = conf[RIGHT_SHOULDER]

        if lsc >= MIN_CONF and rsc >= MIN_CONF:
            shoulder_center = (ls + rs) / 2
        elif lsc >= MIN_CONF:
            shoulder_center = ls
        elif rsc >= MIN_CONF:
            shoulder_center = rs
        else:
            shoulder_center = hip_center

        body_size = np.linalg.norm(shoulder_center - hip_center)
        if body_size < 1e-6:
            body_size = 1.0

        # Chuẩn hóa: trừ tâm hông, chia body_size
        xy_normalized = (xy - hip_center) / body_size
        xy_normalized[conf < MIN_CONF] = 0
        skeleton[:, :2] = xy_normalized

        return skeleton

    # ============================================================
    #  PERSON TRACKING (IoU-based)
    # ============================================================
    def compute_iou(self, box1, box2):
        """Tính IoU giữa 2 bounding box"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def track_persons(self, detections):
        """Tracking người qua các frame bằng IoU matching"""
        matched = {}
        new_tracked = {}
        used_det = set()

        # Match existing tracks
        for tid, tbox in self.tracked_bboxes.items():
            best_iou = 0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in used_det:
                    continue
                iou = self.compute_iou(tbox, det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= self.iou_threshold and best_idx >= 0:
                matched[tid] = best_idx
                new_tracked[tid] = detections[best_idx][:4]
                used_det.add(best_idx)

        # Assign new IDs for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_det:
                new_tracked[self.next_id] = det[:4]
                matched[self.next_id] = i
                self.next_id += 1

        self.tracked_bboxes = new_tracked
        return matched

    # ============================================================
    #  TCN PREDICTION
    # ============================================================
    def predict(self, buffer):
        """Dự đoán xác suất ngã từ buffer 30 frames"""
        if len(buffer) < CONFIG['window_size']:
            return 0.0

        window = np.array(buffer)
        window_flat = window.reshape(window.shape[0], -1)

        input_tensor = torch.FloatTensor(window_flat).unsqueeze(0)
        input_tensor = input_tensor.to(CONFIG['device'])

        # Tính Velocity Features ngay trong lúc inference
        input_tensor = add_velocity_features(input_tensor)

        with torch.no_grad():
            prob = self.model(input_tensor).item()

        return prob

    # ============================================================
    #  DRAWING & UI
    # ============================================================
    def draw_skeleton(self, frame, skeleton):
        """Vẽ khung xương lên frame"""
        for idx in range(min(17, len(skeleton))):
            x, y, conf = skeleton[idx]
            if conf > MIN_CONF:
                cv2.circle(frame, (int(x), int(y)), 3, COLORS['skeleton'], -1)

        for start, end in SKELETON_CONNECTIONS:
            if start < len(skeleton) and end < len(skeleton):
                if skeleton[start, 2] > MIN_CONF and skeleton[end, 2] > MIN_CONF:
                    p1 = (int(skeleton[start, 0]), int(skeleton[start, 1]))
                    p2 = (int(skeleton[end, 0]), int(skeleton[end, 1]))
                    cv2.line(frame, p1, p2, COLORS['skeleton'], 2)

    def draw_person_label(self, frame, person_id, bbox, status, prob):
        """Vẽ nhãn trạng thái cho từng người"""
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        color = COLORS['fall'] if status == "FALL" else COLORS['normal']

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"ID:{person_id} {status} ({prob*100:.0f}%)"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = max(y1 - 10, label_size[1] + 5)

        cv2.rectangle(frame,
                      (x1, label_y - label_size[1] - 5),
                      (x1 + label_size[0] + 5, label_y + 5),
                      color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_white'], 2)

    def draw_info_panel(self, frame, fps):
        """Vẽ bảng thông tin tổng hợp ở góc trên bên trái"""
        h, w = frame.shape[:2]
        any_fall = any(s == "FALL" for s in self.statuses.values())
        num_persons = len(self.statuses)

        # Panel background
        panel_h = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, panel_h), COLORS['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "FALL DETECTION v2", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_white'], 2)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_white'], 1)

        # Persons count
        cv2.putText(frame, f"Persons: {num_persons}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_white'], 1)

        # System status
        if any_fall:
            status_color = COLORS['fall']
            status_text = "STATUS: FALL DETECTED!"
        else:
            status_color = COLORS['normal']
            status_text = "STATUS: NORMAL"

        cv2.putText(frame, status_text, (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    def draw_fall_alert(self, frame):
        """Vẽ cảnh báo FALL toàn màn hình"""
        h, w = frame.shape[:2]
        any_fall = any(s == "FALL" for s in self.statuses.values())

        if any_fall:
            # Viền đỏ xung quanh
            cv2.rectangle(frame, (3, 3), (w - 3, h - 3), COLORS['fall'], 4)

            # Banner cảnh báo
            alert_text = "!! FALL DETECTED !!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 40

            cv2.rectangle(frame,
                          (text_x - 15, text_y - text_size[1] - 15),
                          (text_x + text_size[0] + 15, text_y + 15),
                          COLORS['fall'], -1)
            cv2.putText(frame, alert_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['text_white'], 3)

    # ============================================================
    #  MAIN PROCESSING LOOP
    # ============================================================
    def process_frame(self, frame):
        """Xử lý 1 frame: Detect → Track → Normalize → Predict → Draw"""

        # Bước 1: Phát hiện người + trích xuất keypoints
        detections, keypoints_dict = self.detect_persons(frame)

        if detections:
            # Bước 2: Tracking người qua các frame
            matched = self.track_persons(detections)

            # Dọn dẹp người đã biến mất khỏi khung hình
            active_ids = set(matched.keys())
            for pid in list(self.statuses.keys()):
                if pid not in active_ids:
                    for store in [self.statuses, self.fall_probabilities,
                                  self.skeleton_buffers, self.fall_alert_timers]:
                        store.pop(pid, None)

            # Xử lý từng người
            for person_id, det_idx in matched.items():
                bbox = detections[det_idx]

                # Khởi tạo buffer nếu là người mới
                if person_id not in self.skeleton_buffers:
                    self.skeleton_buffers[person_id] = deque(maxlen=CONFIG['window_size'])
                    self.fall_probabilities[person_id] = 0.0
                    self.statuses[person_id] = "NORMAL"
                    self.fall_alert_timers[person_id] = 0

                buffer = self.skeleton_buffers[person_id]

                # Countdown timer cảnh báo
                if self.fall_alert_timers.get(person_id, 0) > 0:
                    self.fall_alert_timers[person_id] -= 1

                # Lấy skeleton gốc (chưa chuẩn hóa) để vẽ lên frame
                skeleton_raw = keypoints_dict.get(det_idx)

                if skeleton_raw is not None:
                    # Vẽ khung xương lên frame
                    self.draw_skeleton(frame, skeleton_raw)

                    # Bước 3: Chuẩn hóa skeleton
                    skeleton_norm = self.normalize_skeleton(skeleton_raw)

                    if skeleton_norm is not None:
                        # Đẩy vào buffer (Sliding Window)
                        buffer.append(skeleton_norm)

                        # Bước 4: Khi buffer đủ 30 frames → Dự đoán
                        if len(buffer) >= CONFIG['window_size']:
                            prob = self.predict(buffer)
                            self.fall_probabilities[person_id] = prob

                            # BO LOC CHONG NHIEU 5 FRAMES + TU DONG LUU ANH
                            if prob >= CONFIG['threshold']:
                                if not hasattr(self, 'fall_consecutive_frames'):
                                    self.fall_consecutive_frames = {}
                                self.fall_consecutive_frames[person_id] = self.fall_consecutive_frames.get(person_id, 0) + 1
                            else:
                                if hasattr(self, 'fall_consecutive_frames'):
                                    self.fall_consecutive_frames[person_id] = 0
                                    
                            if getattr(self, 'fall_consecutive_frames', {}).get(person_id, 0) >= 5:
                                if self.statuses.get(person_id) != "FALL":
                                    # Luu anh khoanh khac nga
                                    import os, time
                                    save_dir = "alerts"
                                    os.makedirs(save_dir, exist_ok=True)
                                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                                    filename = os.path.join(save_dir, f"fall_detected_p{person_id}_{timestamp}.jpg")
                                    cv2.imwrite(filename, frame)
                                    print(f"  [ALERT] Phat hien NGA! Da luu anh: {filename}")
                                
                                self.statuses[person_id] = "FALL"
                                self.fall_alert_timers[person_id] = self.fall_alert_duration
                            elif self.fall_alert_timers.get(person_id, 0) <= 0:
                                self.statuses[person_id] = "NORMAL"

                # Vẽ nhãn trạng thái cho người này
                status = self.statuses.get(person_id, "NORMAL")
                prob = self.fall_probabilities.get(person_id, 0.0)
                self.draw_person_label(frame, person_id, bbox, status, prob)

        # Vẽ cảnh báo toàn màn hình nếu có người ngã
        self.draw_fall_alert(frame)

        return frame

    def run(self, source=0):
        """
        Chạy hệ thống phát hiện ngã

        Args:
            source: 0 = webcam, hoặc đường dẫn file video
        """
        # Mở camera/video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Khong the mo nguon video: {source}")
            return

        is_webcam = isinstance(source, int)

        # Thiết lập camera nếu là webcam
        if is_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera_width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera_height'])
            cap.set(cv2.CAP_PROP_FPS, CONFIG['camera_fps'])
            print(f"\nWebcam: {CONFIG['camera_width']}x{CONFIG['camera_height']} @ {CONFIG['camera_fps']}FPS")
        else:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"\nVideo: {w}x{h} @ {fps}FPS, {total} frames ({total/fps:.1f}s)")

        print()
        print("-" * 65)
        print("  Phim tat:")
        print("    [q] Thoat")
        print("    [s] Chup man hinh")
        print("    [r] Reset toan bo tracker")
        print("    [SPACE] Tam dung / Tiep tuc")
        print("-" * 65)
        print()

        paused = False
        start_time = time.time()

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        if is_webcam:
                            print("ERROR: Khong doc duoc frame tu webcam!")
                        else:
                            print("\nHet video.")
                        break

                    self.frame_count += 1

                    # Tính FPS
                    self.fps_history.append(time.time())
                    if len(self.fps_history) >= 2:
                        fps = len(self.fps_history) / (self.fps_history[-1] - self.fps_history[0])
                    else:
                        fps = 0

                    # Xử lý frame
                    processed = self.process_frame(frame)

                    # Vẽ bảng thông tin
                    self.draw_info_panel(processed, fps)

                    # Hiển thị
                    cv2.imshow("Fall Detection System v2", processed)

                # Xử lý phím bấm
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nDang thoat...")
                    break
                elif key == ord('s'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{ts}.jpg"
                    cv2.imwrite(filename, processed)
                    print(f"Da luu anh: {filename}")
                elif key == ord('r'):
                    self.skeleton_buffers.clear()
                    self.fall_probabilities.clear()
                    self.statuses.clear()
                    self.tracked_bboxes.clear()
                    self.fall_alert_timers.clear()
                    self.next_id = 0
                    print("Da reset toan bo tracker!")
                elif key == ord(' '):
                    paused = not paused
                    print("TAM DUNG" if paused else "TIEP TUC")

        except KeyboardInterrupt:
            print("\n\nNhan Ctrl+C, dang thoat...")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            elapsed = time.time() - start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

            print()
            print("=" * 65)
            print("  KET QUA")
            print("=" * 65)
            print(f"  Tong so frame: {self.frame_count}")
            print(f"  Thoi gian chay: {elapsed:.1f}s")
            print(f"  FPS trung binh: {avg_fps:.1f}")
            print("=" * 65)


# ============================================================
#  ENTRY POINT
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Fall Detection System v2 - YOLOv8-Pose + FallTCN (Velocity)'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='0',
        help='Nguon video: 0 = webcam (mac dinh), hoac duong dan file video'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Nguong phat hien nga (mac dinh: 0.5)'
    )

    args = parser.parse_args()
    CONFIG['threshold'] = args.threshold

    # Tự động nhận diện webcam index hoặc file path
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    system = FallDetectionSystemV2()
    system.run(source=source)


if __name__ == "__main__":
    main()
