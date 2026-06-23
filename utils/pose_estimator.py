"""
Module Pose Estimation sử dụng YOLOv8-pose
Thay thế MediaPipe bằng YOLOv8-pose (COCO 17 keypoints)
"""
import cv2
import numpy as np
from ultralytics import YOLO

# COCO index → MediaPipe index mapping (cho FallDetector tương thích)
COCO_TO_MP = {0: 0, 5: 11, 6: 12, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}

class PoseEstimator:
    def __init__(self, model='yolov8m-pose.pt'):
        self.yolo = YOLO(model)

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        results = self.yolo(frame, conf=0.5, verbose=False)
        landmarks = None
        bbox_normalized = None
        bbox_pixel = None

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            if len(result.boxes) == 0:
                continue

            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            kps_xy = result.keypoints.xy[0].cpu().numpy()
            kps_conf = result.keypoints.conf[0].cpu().numpy()

            # Tạo list landMark giả MediaPipe (33 pt) để FallDetector không lỗi
            landmarks = [None] * 33
            for ci, mi in COCO_TO_MP.items():
                lm = type('Landmark', (), {
                    'x': float(kps_xy[ci, 0]) / w,
                    'y': float(kps_xy[ci, 1]) / h,
                    'z': 0.0,
                    'visibility': float(kps_conf[ci])
                })()
                landmarks[mi] = lm

            # Kiểm tra các điểm quan trọng có tồn tại không
            if landmarks[11] is None or landmarks[12] is None:
                landmarks = None
                continue

            padding = 0.05
            xs = [kps_xy[i, 0] / w for i in range(17) if kps_conf[i] > 0.3]
            ys = [kps_xy[i, 1] / h for i in range(17) if kps_conf[i] > 0.3]
            if xs and ys:
                bbox_normalized = [
                    max(0, min(xs) - padding),
                    max(0, min(ys) - padding),
                    min(1, max(xs) + padding),
                    min(1, max(ys) + padding)
                ]
                bbox_pixel = [
                    int(bbox_normalized[0] * w),
                    int(bbox_normalized[1] * h),
                    int(bbox_normalized[2] * w),
                    int(bbox_normalized[3] * h)
                ]
            else:
                landmarks = None

        return landmarks, bbox_normalized, bbox_pixel

    def draw_landmarks(self, frame, landmarks):
        if landmarks is None:
            return frame
        h, w = frame.shape[:2]

        # COCO connections để vẽ
        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        COCO_IDX = {0: 0, 5: 11, 6: 12, 7: 7, 8: 8, 9: 9, 10: 10,
                    11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}

        # Vẽ điểm
        for ci, mi in COCO_IDX.items():
            if mi < len(landmarks) and landmarks[mi] is not None and landmarks[mi].visibility > 0.3:
                x = int(landmarks[mi].x * w)
                y = int(landmarks[mi].y * h)
                cv2.circle(frame, (x, y), 3, (255, 200, 100), -1)

        # Vẽ đường nối
        for start, end in connections:
            sm = COCO_IDX.get(start)
            em = COCO_IDX.get(end)
            if sm is not None and em is not None:
                if (sm < len(landmarks) and landmarks[sm] is not None
                        and em < len(landmarks) and landmarks[em] is not None):
                    if landmarks[sm].visibility > 0.3 and landmarks[em].visibility > 0.3:
                        p1 = (int(landmarks[sm].x * w), int(landmarks[sm].y * h))
                        p2 = (int(landmarks[em].x * w), int(landmarks[em].y * h))
                        cv2.line(frame, p1, p2, (255, 200, 100), 2)
        return frame

    def close(self):
        pass
