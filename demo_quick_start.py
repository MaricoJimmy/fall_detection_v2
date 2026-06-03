"""
Demo nhanh - Fall Detection System
Phien ban don gian, de su dung cho demo nhanh
"""
import cv2
import time
from datetime import datetime
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import os


class QuickFallDetectionDemo:
    """
    Demo nhanh he thong phat hien nga
    """

    def __init__(self):
        print("=" * 70)
        print("  FALL DETECTION DEMO - AI-Based Human Fall Detection System")
        print("  Using: YOLOv8 + MediaPipe Pose Estimation")
        print("=" * 70)
        print()

        # Load YOLOv8
        print("[1] Loading YOLOv8...")
        self.yolo = YOLO('yolov8n.pt')
        print("    Done!")

        # Load MediaPipe Pose with new Tasks API
        print("[2] Loading MediaPipe Pose...")
        try:
            # Download pose model if not exists
            model_path = 'pose_landmarker_lite.task'
            if not os.path.exists(model_path):
                print("    Downloading pose model...")
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                urllib.request.urlretrieve(url, model_path)
                print("    Model downloaded!")
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"    Warning: Using fallback mode - {e}")
            self.detector = None
        
        print("    Done!")

        # State tracking
        self.fall_start_time = None
        self.is_falling = False

        print()
        print("Ready! Press 'q' to quit, 'r' to reset")
        print("-" * 70)

    def run(self):
        """Run demo"""
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Step 1: Detect person with YOLO
            results = self.yolo(frame, conf=0.5, classes=[0], verbose=False)

            fall_detected = False
            angle = 0
            aspect_ratio = 1.0

            # Step 2: If person detected, do pose estimation
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                 (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}",
                               (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Crop person region
                    padding = 20
                    h, w = frame.shape[:2]
                    x1_c = max(0, int(x1) - padding)
                    y1_c = max(0, int(y1) - padding)
                    x2_c = min(w, int(x2) + padding)
                    y2_c = min(h, int(y2) + padding)

                    person_roi = frame[y1_c:y2_c, x1_c:x2_c]

                    if person_roi.size > 0 and self.detector:
                        # Pose estimation with new API
                        try:
                            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                            mp_image = python.Image(
                                image_format=vision.ImageFormat.SRGB,
                                data=rgb_roi
                            )
                            detection_result = self.detector.detect(mp_image)
                            
                            if detection_result.pose_landmarks:
                                landmarks_list = detection_result.pose_landmarks[0]
                                
                                # Draw skeleton
                                pose_connections = [
                                    (0, 1), (0, 4), (1, 2), (2, 3), (1, 5), (5, 6), (6, 8),
                                    (5, 7), (7, 9), (4, 5), (9, 10), (11, 12), (11, 13),
                                    (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
                                    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
                                    (27, 29), (28, 30), (29, 31), (30, 32), (27, 28),
                                    (28, 32), (29, 30)
                                ]
                                
                                for start_idx, end_idx in pose_connections:
                                    if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
                                        start = landmarks_list[start_idx]
                                        end = landmarks_list[end_idx]
                                        
                                        if start.presence > 0.5 and end.presence > 0.5:
                                            start_pt = (int(start.x * (x2_c - x1_c) + x1_c),
                                                       int(start.y * (y2_c - y1_c) + y1_c))
                                            end_pt = (int(end.x * (x2_c - x1_c) + x1_c),
                                                     int(end.y * (y2_c - y1_c) + y1_c))
                                            cv2.line(frame, start_pt, end_pt, (255, 200, 100), 2)

                                # Draw keypoints
                                for lm in landmarks_list:
                                    if lm.presence > 0.5:
                                        x = int(lm.x * (x2_c - x1_c) + x1_c)
                                        y = int(lm.y * (y2_c - y1_c) + y1_c)
                                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

                                # Calculate body angle
                                if len(landmarks_list) > 24:
                                    left_shoulder = landmarks_list[11]
                                    right_shoulder = landmarks_list[12]
                                    left_hip = landmarks_list[23]
                                    right_hip = landmarks_list[24]

                                    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                                    hip_center_y = (left_hip.y + right_hip.y) / 2

                                    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                                    hip_center_x = (left_hip.x + right_hip.x) / 2

                                    dx = shoulder_center_x - hip_center_x
                                    dy = shoulder_center_y - hip_center_y

                                    angle = abs(90 - abs(180 / 3.14159 * (3.14159 / 2 +
                                             0 if dy == 0 else 3.14159 / 4 * (dx / abs(dy)))))

                                    # Aspect ratio
                                    bbox_width = x2 - x1
                                    bbox_height = y2 - y1
                                    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1

                                    # Fall detection logic
                                    fall_detected = (angle > 45 and aspect_ratio > 1.2)

                                    # Track fall duration
                                    current_time = time.time()
                                    if fall_detected:
                                        if not self.is_falling:
                                            self.is_falling = True
                                            self.fall_start_time = current_time
                                        time_falling = current_time - self.fall_start_time

                                        if time_falling > 1.0:
                                            # CONFIRMED FALL
                                            cv2.rectangle(frame, (5, 5),
                                                       (frame.shape[1]-5, frame.shape[0]-5),
                                                       (0, 0, 255), 5)
                                            cv2.putText(frame,
                                                       "!!! CANH BAO: PHAT HIEN NGA !!!",
                                                       (frame.shape[1]//2 - 180, 50),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                       (0, 0, 255), 3)
                                            cv2.putText(frame,
                                                       f"Thoi gian: {time_falling:.1f}s",
                                                       (20, frame.shape[0] - 60),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                       (0, 0, 255), 2)

                                            # Alert beep
                                            try:
                                                import winsound
                                                winsound.Beep(1000, 200)
                                            except:
                                                pass
                                    else:
                                        self.is_falling = False
                                        self.fall_start_time = None
                        except Exception as e:
                            print(f"Pose detection error: {e}")
                            pass

            # Draw info panel
            info_panel = [
                f"Body Angle: {angle:.1f} deg",
                f"Aspect Ratio: {aspect_ratio:.2f}",
                f"Status: {'FALL' if fall_detected and self.is_falling else 'NORMAL'}",
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            ]

            for i, text in enumerate(info_panel):
                cv2.putText(frame, text, (frame.shape[1] - 200, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Fall Detection Demo - YOLOv8 + MediaPipe", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('r'):
                self.is_falling = False
                self.fall_start_time = None
                print("Reset!")

        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo ended!")


if __name__ == "__main__":
    demo = QuickFallDetectionDemo()
    demo.run()