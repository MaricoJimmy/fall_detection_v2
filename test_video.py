"""
Script test model TCN tren video thuc te
Su dung model da train de phat hien nga tren video moi
"""
import os
import sys
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
import time
from datetime import datetime

# Import model TCN
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_tcn_only import FallTCN

# Cau hinh
CONFIG = {
    'model_path': 'models/skeleton/TCN_best.pth',
    'yolo_model': 'yolov8n.pt',
    'window_size': 30,
    'num_features': 51,
    'threshold': 0.5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Thu muc luu output
OUTPUT_DIR = 'test_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 17 keypoints COCO format
KEYPOINT_NAMES = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

class FallDetectionTester:
    def __init__(self):
        print("="*70)
        print("  FALL DETECTION - VIDEO TESTER (TCN MODEL)")
        print("="*70)
        print(f"Device: {CONFIG['device']}")
        if CONFIG['device'] == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load TCN model
        print("\n[1/3] Loading TCN model...")
        self.model = FallTCN(
            input_size=CONFIG['num_features'],
            num_channels=[64, 128, 128],
            kernel_size=3,
            dropout=0.3
        )
        checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'], weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(CONFIG['device'])
        self.model.eval()
        print("      ✓ TCN model loaded")
        
        # Load YOLO
        print("[2/3] Loading YOLO model...")
        self.yolo = YOLO(CONFIG['yolo_model'])
        print("      ✓ YOLO model loaded")
        
        # Load MediaPipe
        print("[3/3] Loading MediaPipe...")
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("      ✓ MediaPipe loaded")
        
        # Buffer de luu 30 frames gan nhat
        self.skeleton_buffer = deque(maxlen=CONFIG['window_size'])
        
        # Ket qua
        self.fall_probability = 0.0
        self.status = "NORMAL"
        
        print("\n" + "="*70)
        print("  SAN SANG TEST VIDEO")
        print("="*70)
    
    def detect_person(self, frame):
        """Phat hien nguoi bang YOLO"""
        results = self.yolo(frame, conf=0.5, classes=[0], verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])
        
        return detections
    
    def extract_skeleton(self, frame, bbox):
        """Trich xuat skeleton tu MediaPipe"""
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        h, w = frame.shape[:2]
        
        # Crop person region
        padding = 20
        x1_crop = max(0, x1 - padding)
        y1_crop = max(0, y1 - padding)
        x2_crop = min(w, x2 + padding)
        y2_crop = min(h, y2 + padding)
        
        person_region = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if person_region.size == 0:
            return None
        
        # Convert to RGB
        rgb = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.mp_pose.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Trich xuat 17 keypoints
        skeleton = np.full((17, 3), np.nan)
        
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx < 17:
                skeleton[idx] = [
                    landmark.x * (x2_crop - x1_crop),
                    landmark.y * (y2_crop - y1_crop),
                    landmark.visibility
                ]
        
        return skeleton
    
    def normalize_skeleton(self, skeleton):
        """Chuan hoa skeleton giong nhu khi train"""
        if skeleton is None:
            return None
        
        xy = skeleton[:, :2]
        conf = skeleton[:, 2]
        
        # Tinh hip center
        left_hip = xy[LEFT_HIP]
        right_hip = xy[RIGHT_HIP]
        left_hip_conf = conf[LEFT_HIP]
        right_hip_conf = conf[RIGHT_HIP]
        
        if left_hip_conf < 0.3 and right_hip_conf < 0.3:
            return None
        
        if left_hip_conf >= 0.3 and right_hip_conf >= 0.3:
            hip_center = (left_hip + right_hip) / 2
        elif left_hip_conf >= 0.3:
            hip_center = left_hip
        else:
            hip_center = right_hip
        
        # Tinh body size
        left_shoulder = xy[LEFT_SHOULDER]
        right_shoulder = xy[RIGHT_SHOULDER]
        left_shoulder_conf = conf[LEFT_SHOULDER]
        right_shoulder_conf = conf[RIGHT_SHOULDER]
        
        if left_shoulder_conf >= 0.3 and right_shoulder_conf >= 0.3:
            shoulder_center = (left_shoulder + right_shoulder) / 2
        elif left_shoulder_conf >= 0.3:
            shoulder_center = left_shoulder
        elif right_shoulder_conf >= 0.3:
            shoulder_center = right_shoulder
        else:
            shoulder_center = hip_center
        
        body_size = np.linalg.norm(shoulder_center - hip_center)
        if body_size < 1e-6:
            body_size = 1.0
        
        # Chuan hoa
        xy_normalized = (xy - hip_center) / body_size
        
        # Xu ly diem co confidence thap
        low_conf_mask = conf < 0.3
        xy_normalized[low_conf_mask] = 0
        
        skeleton[:, :2] = xy_normalized
        
        return skeleton
    
    def predict(self):
        """Du doan tu buffer 30 frames"""
        if len(self.skeleton_buffer) < CONFIG['window_size']:
            return 0.0
        
        # Tao input (30, 51)
        window = np.array(self.skeleton_buffer)  # (30, 17, 3)
        window_flat = window.reshape(window.shape[0], -1)  # (30, 51)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(window_flat).unsqueeze(0)  # (1, 30, 51)
        input_tensor = input_tensor.to(CONFIG['device'])
        
        # Predict
        with torch.no_grad():
            prob = self.model(input_tensor).item()
        
        return prob
    
    def process_frame(self, frame):
        """Xu ly mot frame"""
        h, w = frame.shape[:2]
        
        # Detect person
        detections = self.detect_person(frame)
        
        if detections:
            # Lay detection dau tien
            bbox = detections[0]
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            
            # Ve bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {bbox[4]:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Extract skeleton
            skeleton = self.extract_skeleton(frame, bbox)
            
            if skeleton is not None:
                # Normalize
                skeleton_norm = self.normalize_skeleton(skeleton)
                
                if skeleton_norm is not None:
                    # Them vao buffer
                    self.skeleton_buffer.append(skeleton_norm)
                    
                    # Predict neu du 30 frames
                    if len(self.skeleton_buffer) >= CONFIG['window_size']:
                        self.fall_probability = self.predict()
                        
                        # Xac dinh trang thai
                        if self.fall_probability >= CONFIG['threshold']:
                            self.status = "FALL"
                        else:
                            self.status = "NORMAL"
                    
                    # Ve skeleton
                    self.draw_skeleton(frame, skeleton, x1, y1)
        
        # Ve ket qua
        self.draw_results(frame)
        
        return frame
    
    def draw_skeleton(self, frame, skeleton, offset_x, offset_y):
        """Ve skeleton len frame"""
        connections = [
            (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (24, 26), (25, 27), (26, 28)
        ]
        
        # Ve diem
        for idx in range(min(17, len(skeleton))):
            x, y, conf = skeleton[idx]
            if conf > 0.3:
                px = int(x + offset_x)
                py = int(y + offset_y)
                cv2.circle(frame, (px, py), 3, (255, 200, 100), -1)
        
        # Ve duong noi
        for start, end in connections:
            if start < len(skeleton) and end < len(skeleton):
                if skeleton[start, 2] > 0.3 and skeleton[end, 2] > 0.3:
                    p1 = (int(skeleton[start, 0] + offset_x), int(skeleton[start, 1] + offset_y))
                    p2 = (int(skeleton[end, 0] + offset_x), int(skeleton[end, 1] + offset_y))
                    cv2.line(frame, p1, p2, (255, 200, 100), 2)
    
    def draw_results(self, frame):
        """Ve ket qua len frame"""
        h, w = frame.shape[:2]
        
        # Background panel
        panel_w = 350
        panel_h = 180
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "FALL DETECTION (TCN)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Status
        if self.status == "FALL":
            status_color = (0, 0, 255)  # Red
            status_text = "FALL DETECTED!"
        else:
            status_color = (0, 255, 0)  # Green
            status_text = "NORMAL"
        
        cv2.putText(frame, f"Status: {status_text}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Probability
        prob_percent = self.fall_probability * 100
        cv2.putText(frame, f"Fall Probability: {prob_percent:.1f}%", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Buffer size
        buffer_size = len(self.skeleton_buffer)
        cv2.putText(frame, f"Buffer: {buffer_size}/{CONFIG['window_size']} frames", (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (20, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Alert neu FALL
        if self.status == "FALL":
            # Ve vien do
            cv2.rectangle(frame, (5, 5), (w-5, h-5), (0, 0, 255), 5)
            
            # Alert text
            alert_text = "WARNING: FALL DETECTED!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            cv2.rectangle(frame, (text_x-10, text_y-40), (text_x+text_size[0]+10, text_y+10),
                         (0, 0, 255), -1)
            cv2.putText(frame, alert_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    def test_video(self, video_path, output_path=None):
        """Test tren video"""
        print(f"\nOpening video: {video_path}")
        
        # Tao subfolder cho moi video test
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(OUTPUT_DIR, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video {video_path}")
            return
        
        # Video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f}s")
        
        # Video writer - luu vao test_outputs/<video_name>/
        out = None
        if output_path is None:
            # Tu dong tao ten file output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(video_output_dir, f"output_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"\nSaving output video to: {output_path}")
        print(f"All outputs will be saved in: {video_output_dir}")
        
        print(f"\n{'='*70}")
        print("  PRESS 'q' TO QUIT, 's' TO SAVE SCREENSHOT")
        print(f"{'='*70}\n")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video")
                    break
                
                frame_count += 1
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Hien thi
                cv2.imshow("Fall Detection Test", processed_frame)
                
                # Save video
                if out:
                    out.write(processed_frame)
                
                # Progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - Status: {self.status} - Prob: {self.fall_probability*100:.1f}%")
                
                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuit by user")
                    break
                elif key == ord('s'):
                    # Luu screenshot vao test_outputs/<video_name>/
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(video_output_dir, f"screenshot_{timestamp}.jpg")
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            
            print(f"\n{'='*70}")
            print("  TEST COMPLETE")
            print(f"{'='*70}")
            print(f"Frames processed: {frame_count}")
            print(f"Time elapsed: {elapsed:.1f}s")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Final status: {self.status}")
            print(f"Final probability: {self.fall_probability*100:.1f}%")
            print(f"\nAll outputs saved in: {video_output_dir}")
            print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test TCN model on video')
    parser.add_argument('--video', '-v', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save output video (optional)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Fall detection threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    CONFIG['threshold'] = args.threshold
    
    tester = FallDetectionTester()
    tester.test_video(args.video, args.output)

if __name__ == '__main__':
    main()
