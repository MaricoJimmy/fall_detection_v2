"""
Script phan tich chi tiet tung frame de debug
Luu log va video voi thong tin chi tiet hon
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
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_tcn_only import FallTCN

CONFIG = {
    'model_path': 'models/skeleton/TCN_best.pth',
    'yolo_model': 'yolov8n.pt',
    'window_size': 30,
    'num_features': 51,
    'threshold': 0.5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

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

class DetailedAnalyzer:
    def __init__(self):
        print("="*70)
        print("  DETAILED FALL ANALYSIS")
        print("="*70)
        
        # Load models
        print("\nLoading models...")
        self.model = FallTCN(input_size=51, num_channels=[64, 128, 128], kernel_size=3, dropout=0.3)
        checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'], weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(CONFIG['device'])
        self.model.eval()
        
        self.yolo = YOLO(CONFIG['yolo_model'])
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.skeleton_buffer = deque(maxlen=CONFIG['window_size'])
        self.log_data = []
        
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
    
    def predict(self):
        if len(self.skeleton_buffer) < CONFIG['window_size']:
            return 0.0
        
        window = np.array(self.skeleton_buffer)
        window_flat = window.reshape(window.shape[0], -1)
        
        input_tensor = torch.FloatTensor(window_flat).unsqueeze(0)
        input_tensor = input_tensor.to(CONFIG['device'])
        
        with torch.no_grad():
            prob = self.model(input_tensor).item()
        
        return prob
    
    def analyze_video(self, video_path):
        print(f"Analyzing video: {video_path}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {total_frames} frames, {fps} FPS, {total_frames/fps:.1f}s")
        print(f"Analyzing...\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Detect person
            detections = self.detect_person(frame)
            
            yolo_detected = len(detections) > 0
            skeleton_extracted = False
            skeleton_normalized = False
            fall_prob = 0.0
            status = "NORMAL"
            
            if yolo_detected:
                bbox = detections[0]
                skeleton = self.extract_skeleton(frame, bbox)
                
                if skeleton is not None:
                    skeleton_extracted = True
                    skeleton_norm = self.normalize_skeleton(skeleton)
                    
                    if skeleton_norm is not None:
                        skeleton_normalized = True
                        self.skeleton_buffer.append(skeleton_norm)
                        
                        if len(self.skeleton_buffer) >= CONFIG['window_size']:
                            fall_prob = self.predict()
                            status = "FALL" if fall_prob >= CONFIG['threshold'] else "NORMAL"
            
            # Log data
            log_entry = {
                'frame': frame_count,
                'timestamp': f"{timestamp:.2f}",
                'yolo_detected': yolo_detected,
                'skeleton_extracted': skeleton_extracted,
                'skeleton_normalized': skeleton_normalized,
                'buffer_size': len(self.skeleton_buffer),
                'fall_probability': f"{fall_prob*100:.1f}",
                'status': status
            }
            
            self.log_data.append(log_entry)
            
            # Print important frames
            if frame_count % 30 == 0 or fall_prob > 0.3:
                print(f"Frame {frame_count:3d} ({timestamp:5.2f}s) | "
                      f"YOLO: {'✓' if yolo_detected else '✗'} | "
                      f"Skeleton: {'✓' if skeleton_extracted else '✗'} | "
                      f"Buffer: {len(self.skeleton_buffer):2d}/{CONFIG['window_size']} | "
                      f"Prob: {fall_prob*100:5.1f}% | "
                      f"Status: {status}")
        
        cap.release()
        
        # Save log to CSV
        csv_path = 'frame_analysis_log.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writeheader()
            writer.writerows(self.log_data)
        
        print(f"\n{'='*70}")
        print("  ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Total frames: {frame_count}")
        print(f"Log saved to: {csv_path}")
        
        # Analyze critical moments
        print(f"\n{'='*70}")
        print("  CRITICAL MOMENTS")
        print(f"{'='*70}")
        
        # Tim frame co probability cao
        high_prob_frames = [log for log in self.log_data if float(log['fall_probability']) > 30]
        if high_prob_frames:
            print(f"\nFrames with Fall Probability > 30%:")
            for log in high_prob_frames:
                print(f"  Frame {log['frame']} ({log['timestamp']}s): {log['fall_probability']}%")
        
        # Tim frame khong detect duoc nguoi
        no_detect_frames = [log for log in self.log_data if not log['yolo_detected']]
        if no_detect_frames:
            print(f"\nFrames without person detection: {len(no_detect_frames)}")
            print(f"  First: Frame {no_detect_frames[0]['frame']} ({no_detect_frames[0]['timestamp']}s)")
            print(f"  Last: Frame {no_detect_frames[-1]['frame']} ({no_detect_frames[-1]['timestamp']}s)")
        
        # Phan tich doan 5 giay (frame ~145)
        print(f"\n{'='*70}")
        print("  ANALYSIS AT 5 SECONDS (Expected Fall)")
        print(f"{'='*70}")
        
        target_frame = int(5 * fps)
        start = max(0, target_frame - 30)
        end = min(frame_count, target_frame + 30)
        
        print(f"\nFrames {start}-{end} (around 5 seconds):")
        for log in self.log_data[start-1:end]:
            frame_num = int(log['frame'])
            if frame_num >= start and frame_num <= end:
                try:
                    ts = float(log['timestamp'])
                    print(f"  Frame {frame_num:3d} ({ts:5.2f}s) | "
                          f"YOLO: {'✓' if log['yolo_detected'] else '✗'} | "
                          f"Skeleton: {'✓' if log['skeleton_extracted'] else '✗'} | "
                          f"Buffer: {log['buffer_size']:2d} | "
                          f"Prob: {log['fall_probability']:5.1f}%")
                except:
                    pass
        
        print(f"\n{'='*70}")
        print("  POSSIBLE ISSUES")
        print(f"{'='*70}")
        
        # Kiem tra cac van de co the
        issues = []
        
        # 1. Buffer chua du 30 frames o doan 5 giay
        frame_at_5s = next((log for log in self.log_data if float(log['timestamp']) >= 5.0), None)
        if frame_at_5s and int(frame_at_5s['buffer_size']) < 30:
            issues.append(f"Buffer chỉ có {frame_at_5s['buffer_size']} frames ở giây thứ 5 (cần 30)")
        
        # 2. YOLO khong detect duoc nguoi
        if len(no_detect_frames) > frame_count * 0.3:
            issues.append(f"YOLO không detect được người ở {len(no_detect_frames)} frames ({len(no_detect_frames)/frame_count*100:.1f}%)")
        
        # 3. Skeleton extraction fail
        no_skeleton = [log for log in self.log_data if log['yolo_detected'] and not log['skeleton_extracted']]
        if len(no_skeleton) > 10:
            issues.append(f"Skeleton extraction thất bại ở {len(no_skeleton)} frames dù YOLO detect được người")
        
        if issues:
            print("\nCác vấn đề phát hiện:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n✓ Không phát hiện vấn đề rõ ràng")
        
        print(f"\n{'='*70}")
        print("  RECOMMENDATIONS")
        print(f"{'='*70}")
        print("\n1. Giảm window size từ 30 xuống 15-20 frames để phát hiện nhanh hơn")
        print("2. Giảm threshold từ 0.5 xuống 0.3-0.4 để nhạy hơn")
        print("3. Kiểm tra chất lượng pose estimation ở đoạn 5 giây")
        print("4. Xem video output để hiểu tại sao model bỏ sót")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Detailed frame analysis')
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to video')
    
    args = parser.parse_args()
    
    analyzer = DetailedAnalyzer()
    analyzer.analyze_video(args.video)

if __name__ == '__main__':
    main()
