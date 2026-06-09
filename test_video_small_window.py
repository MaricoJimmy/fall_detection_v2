"""
Script test voi window size nho (10 frames) va padding strategy
De fix van de buffer khong du frames
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_tcn_only import FallTCN

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

class SmallWindowTester:
    def __init__(self, model_path='models/skeleton/TCN_best.pth'):
        print("="*70)
        print("  SMALL WINDOW + PADDING TEST")
        print("="*70)
        
        # Load models
        print("\nLoading models...")
        self.model = FallTCN(input_size=51, num_channels=[64, 128, 128], kernel_size=3, dropout=0.3)
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to('cuda')
        self.model.eval()
        
        self.yolo = YOLO('yolov8n.pt')
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✓ Models loaded\n")
    
    def detect_person(self, frame):
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
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        h, w = frame.shape[:2]
        
        padding = 20
        x1_crop = max(0, x1 - padding)
        y1_crop = max(0, y1 - padding)
        x2_crop = min(w, x2 + padding)
        y2_crop = min(h, y2 + padding)
        
        person_region = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if person_region.size == 0:
            return None
        
        rgb = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
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
        if skeleton is None:
            return None
        
        xy = skeleton[:, :2]
        conf = skeleton[:, 2]
        
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
        
        xy_normalized = (xy - hip_center) / body_size
        low_conf_mask = conf < 0.3
        xy_normalized[low_conf_mask] = 0
        
        skeleton[:, :2] = xy_normalized
        
        return skeleton
    
    def predict(self, buffer, window_size):
        """Predict voi padding neu buffer khong du"""
        if len(buffer) == 0:
            return 0.0
        
        # Neu buffer chua du, pad bang cach lap frame cuoi
        if len(buffer) < window_size:
            window = list(buffer)
            last_frame = window[-1]
            padding_needed = window_size - len(window)
            window.extend([last_frame] * padding_needed)
        else:
            window = list(buffer)
        
        window = np.array(window)
        window_flat = window.reshape(window.shape[0], -1)
        
        input_tensor = torch.FloatTensor(window_flat).unsqueeze(0)
        input_tensor = input_tensor.to('cuda')
        
        with torch.no_grad():
            prob = self.model(input_tensor).item()
        
        return prob
    
    def test_video(self, video_path, window_size=10, threshold=0.4, use_padding=True):
        """Test video voi window size nho va padding"""
        print(f"Testing video: {video_path}")
        print(f"Window size: {window_size} frames")
        print(f"Threshold: {threshold}")
        print(f"Padding: {'Enabled' if use_padding else 'Disabled'}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {total_frames} frames, {fps} FPS, {total_frames/fps:.1f}s\n")
        
        # Video writer
        output_path = f"test_output_win{window_size}_pad{use_padding}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Saving output to: {output_path}\n")
        print("="*70)
        
        buffer = deque(maxlen=window_size)
        frame_count = 0
        fall_detected_count = 0
        
        target_frame_5s = int(5 * fps)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Detect person
            detections = self.detect_person(frame)
            
            skeleton = None
            if detections:
                bbox = detections[0]
                skeleton = self.extract_skeleton(frame, bbox)
                if skeleton is not None:
                    skeleton = self.normalize_skeleton(skeleton)
            
            # Them vao buffer
            if skeleton is not None:
                buffer.append(skeleton)
            
            # Predict
            prob = 0.0
            if use_padding:
                # Voi padding: predict ngay ca khi buffer chua du
                if len(buffer) > 0:
                    prob = self.predict(buffer, window_size)
            else:
                # Khong padding: chi predict khi buffer du
                if len(buffer) >= window_size:
                    prob = self.predict(buffer, window_size)
            
            status = "FALL" if prob >= threshold else "NORMAL"
            
            if status == "FALL":
                fall_detected_count += 1
            
            # Hien thi ket qua
            cv2.putText(frame, f"Time: {timestamp:.2f}s", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Buffer: {len(buffer)}/{window_size}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Prob: {prob*100:.1f}%", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status
            if status == "FALL":
                cv2.putText(frame, "FALL DETECTED!", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 0, 255), 5)
            else:
                cv2.putText(frame, "NORMAL", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            # Ve skeleton
            if detections and skeleton is not None:
                bbox = detections[0]
                x1, y1 = int(bbox[0]), int(bbox[1])
                
                for i in range(17):
                    if skeleton[i, 2] > 0.3:
                        px = int(skeleton[i, 0] + x1)
                        py = int(skeleton[i, 1] + y1)
                        cv2.circle(frame, (px, py), 3, (255, 200, 100), -1)
            
            out.write(frame)
            
            # In ket qua tai cac thoi diem quan trong
            if frame_count % 30 == 0 or abs(frame_count - target_frame_5s) < 5:
                marker = " <-- TARGET (5s)" if abs(frame_count - target_frame_5s) < 3 else ""
                print(f"Frame {frame_count:3d} ({timestamp:5.2f}s) | "
                      f"Buffer: {len(buffer):2d}/{window_size} | "
                      f"Prob: {prob*100:5.1f}% | "
                      f"Status: {status}{marker}")
        
        cap.release()
        out.release()
        
        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        print(f"Total frames: {frame_count}")
        print(f"FALL detections: {fall_detected_count}")
        print(f"Output saved: {output_path}")
        
        # Ket luan
        print("\n" + "="*70)
        print("CONCLUSIONS")
        print("="*70)
        
        if fall_detected_count > 0:
            print(f"✓ Model detected FALL {fall_detected_count} times")
            print(f"✓ Padding strategy helps detect falls even with incomplete buffer")
        else:
            print(f"✗ No FALL detected")
            print(f"✗ May need smaller window size or lower threshold")
        
        print("\nRecommendations:")
        print(f"1. Window size {window_size} frames ({window_size/fps:.2f}s) is more responsive")
        print(f"2. Padding strategy allows prediction even with missing frames")
        print(f"3. Consider retraining model with window_size={window_size} for better accuracy")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test with small window + padding')
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to video')
    parser.add_argument('--window', '-w', type=int, default=10, help='Window size (default: 10)')
    parser.add_argument('--threshold', '-t', type=float, default=0.4, help='Threshold (default: 0.4)')
    parser.add_argument('--no-padding', action='store_true', help='Disable padding')
    
    args = parser.parse_args()
    
    tester = SmallWindowTester()
    tester.test_video(args.video, 
                     window_size=args.window, 
                     threshold=args.threshold,
                     use_padding=not args.no_padding)

if __name__ == '__main__':
    main()
