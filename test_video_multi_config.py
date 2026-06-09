"""
Script test voi nhieu window size va threshold khac nhau
De tim cau hinh tot nhat cho video nay
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

class MultiConfigTester:
    def __init__(self, model_path='models/skeleton/TCN_best.pth'):
        print("="*70)
        print("  MULTI-CONFIG FALL DETECTION TEST")
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
    
    def predict(self, buffer):
        if len(buffer) < buffer.maxlen:
            return 0.0
        
        window = np.array(buffer)
        window_flat = window.reshape(window.shape[0], -1)
        
        input_tensor = torch.FloatTensor(window_flat).unsqueeze(0)
        input_tensor = input_tensor.to('cuda')
        
        with torch.no_grad():
            prob = self.model(input_tensor).item()
        
        return prob
    
    def test_video(self, video_path, window_sizes=[15, 20, 30], thresholds=[0.3, 0.4, 0.5]):
        """Test video voi nhieu cau hinh"""
        print(f"Testing video: {video_path}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {total_frames} frames, {fps} FPS, {total_frames/fps:.1f}s")
        print(f"\nTesting {len(window_sizes)} window sizes × {len(thresholds)} thresholds = {len(window_sizes)*len(thresholds)} configurations\n")
        
        # Doc tat ca frames va skeletons truoc
        print("Step 1: Extracting all skeletons...")
        all_skeletons = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            detections = self.detect_person(frame)
            
            skeleton = None
            if detections:
                bbox = detections[0]
                skeleton = self.extract_skeleton(frame, bbox)
                if skeleton is not None:
                    skeleton = self.normalize_skeleton(skeleton)
            
            all_skeletons.append(skeleton)
            
            if frame_count % 50 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        print(f"✓ Extracted {len(all_skeletons)} skeletons\n")
        
        # Test tung cau hinh
        print("Step 2: Testing different configurations...\n")
        print("="*70)
        print("RESULTS AT 5 SECONDS (Expected Fall)")
        print("="*70)
        print(f"\n{'Config':<25} {'Prob at 5s':<12} {'Status':<10} {'First FALL':<15}")
        print("-"*70)
        
        target_frame = int(5 * fps)  # Frame ~145
        
        for window_size in window_sizes:
            for threshold in thresholds:
                # Tao buffer
                buffer = deque(maxlen=window_size)
                
                first_fall_frame = None
                prob_at_5s = 0.0
                
                for i, skeleton in enumerate(all_skeletons):
                    if skeleton is not None:
                        buffer.append(skeleton)
                    
                    # Predict khi buffer day
                    if len(buffer) >= window_size:
                        prob = self.predict(buffer)
                        
                        # Ghi nhan probability o frame 5 giay
                        if i == target_frame:
                            prob_at_5s = prob
                        
                        # Ghi nhan frame dau tien bao FALL
                        if prob >= threshold and first_fall_frame is None:
                            first_fall_frame = i
                
                # Hien thi ket qua
                config_name = f"Win={window_size}, Th={threshold}"
                status = "FALL" if prob_at_5s >= threshold else "NORMAL"
                first_fall_str = f"{first_fall_frame/fps:.2f}s" if first_fall_frame else "None"
                
                print(f"{config_name:<25} {prob_at_5s*100:<12.1f}% {status:<10} {first_fall_str:<15}")
        
        print("="*70)
        
        # Phan tich chi tiet
        print("\n" + "="*70)
        print("DETAILED ANALYSIS")
        print("="*70)
        
        # Test voi window_size=15, threshold=0.4 (cau hinh tot nhat)
        print("\nTesting with Window=15, Threshold=0.4 (Recommended):\n")
        
        buffer = deque(maxlen=15)
        print(f"{'Time':<8} {'Frame':<8} {'Prob':<10} {'Status':<10}")
        print("-"*40)
        
        for i, skeleton in enumerate(all_skeletons):
            if skeleton is not None:
                buffer.append(skeleton)
            
            if len(buffer) >= 15 and i % 15 == 0:  # Moi 0.5 giay
                prob = self.predict(buffer)
                status = "FALL" if prob >= 0.4 else "NORMAL"
                timestamp = i / fps
                
                # Chi in nhung doan quan trong
                if 0.5 <= timestamp <= 2.5 or 4.0 <= timestamp <= 6.0 or 10.0 <= timestamp <= 12.0:
                    print(f"{timestamp:<8.2f} {i:<8} {prob*100:<10.1f}% {status:<10}")
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        print("\nBased on analysis:")
        print("1. Window size 15-20 frames (0.5-0.7s) is better for fast falls")
        print("2. Threshold 0.3-0.4 is more sensitive")
        print("3. Consider retraining model with smaller window size")
        print("4. Add temporal smoothing to reduce false positives")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test with multiple configs')
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to video')
    
    args = parser.parse_args()
    
    tester = MultiConfigTester()
    tester.test_video(args.video)

if __name__ == '__main__':
    main()
