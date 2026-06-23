"""
Script so sanh skeleton giua cac doan:
- Giay 5 (miss - khong detect duoc)
- Giay 11 (hit - detect dung)
De hieu tai sao model miss o giay 5
"""
import os
import sys
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
import matplotlib.pyplot as plt

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

class SkeletonAnalyzer:
    def __init__(self, model_path='models/skeleton/TCN_best.pth'):
        print("="*70)
        print("  SKELETON COMPARISON ANALYSIS")
        print("="*70)
        
        # Load models
        print("\nLoading models...")
        self.model = FallTCN(input_size=51, num_channels=[64, 128, 128], kernel_size=3, dropout=0.3)
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(CONFIG['device'])
        self.model.eval()
        
        self.yolo = YOLO('yolov8m-pose.pt')
        
        print("✓ Models loaded\n")
    
    def detect_person(self, frame):
        results = self.yolo(frame, conf=0.5, verbose=False)
        detections = []
        keypoints_list = {}
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
                keypoints_list[len(detections) - 1] = skeleton
        return detections, keypoints_list
    
    def extract_skeleton(self, frame, bbox):
        return None  # YOLOv8-pose provides keypoints in detect_person
    
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
        input_tensor = input_tensor.to(CONFIG['device'])
        
        with torch.no_grad():
            prob = self.model(input_tensor).item()
        
        return prob
    
    def analyze_video(self, video_path, target_times=[1.0, 5.0, 11.0]):
        """Phan tich skeleton tai cac thoi diem cu the"""
        print(f"Analyzing video: {video_path}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {total_frames} frames, {fps} FPS, {total_frames/fps:.1f}s")
        
        # Tinh target frames
        target_frames = {f"{t}s": int(t * fps) for t in target_times}
        print(f"Target frames: {target_frames}\n")
        
        # Doc tat ca frames
        print("Extracting skeletons...")
        all_frames = []
        all_skeletons_raw = []
        all_skeletons_norm = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            all_frames.append(frame)
            
            detections, keypoints_list = self.detect_person(frame)
            
            skeleton_raw = None
            skeleton_norm = None
            
            if detections:
                skeleton_raw = keypoints_list.get(0)
                if skeleton_raw is not None and np.any(skeleton_raw[:, 2] > 0):
                    skeleton_norm = self.normalize_skeleton(skeleton_raw)
            
            all_skeletons_raw.append(skeleton_raw)
            all_skeletons_norm.append(skeleton_norm)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        print(f"✓ Extracted {len(all_skeletons_norm)} skeletons\n")
        
        # Phan tich tung thoi diem
        print("="*70)
        print("SKELETON ANALYSIS AT TARGET TIMES")
        print("="*70)
        
        results = {}
        
        for time_label, target_frame in target_frames.items():
            print(f"\n{'='*70}")
            print(f"TIME: {time_label} (Frame {target_frame})")
            print(f"{'='*70}")
            
            # Lay skeleton tai frame nay
            skeleton_raw = all_skeletons_raw[target_frame]
            skeleton_norm = all_skeletons_norm[target_frame]
            
            if skeleton_raw is None:
                print("❌ No person detected at this frame!")
                results[time_label] = {
                    'detected': False,
                    'skeleton': None,
                    'probability': 0.0
                }
                continue
            
            print("✓ Person detected")
            
            # Thong ke skeleton
            print("\nSkeleton Statistics:")
            print(f"  Visible keypoints: {np.sum(skeleton_raw[:, 2] > 0.3)}/17")
            print(f"  Average confidence: {np.mean(skeleton_raw[:, 2]):.3f}")
            
            # Tinh goc nghien co the
            if skeleton_norm is not None:
                left_shoulder = skeleton_norm[LEFT_SHOULDER, :2]
                right_shoulder = skeleton_norm[RIGHT_SHOULDER, :2]
                left_hip = skeleton_norm[LEFT_HIP, :2]
                right_hip = skeleton_norm[RIGHT_HIP, :2]
                
                shoulder_center = (left_shoulder + right_shoulder) / 2
                hip_center = (left_hip + right_hip) / 2
                
                body_vector = shoulder_center - hip_center
                vertical_vector = np.array([0, -1])
                
                cos_angle = np.dot(body_vector, vertical_vector) / (np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector) + 1e-6)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                
                print(f"  Body angle: {angle:.1f}°")
                
                # Tinh aspect ratio
                x_coords = skeleton_norm[:, 0]
                y_coords = skeleton_norm[:, 1]
                width = np.max(x_coords) - np.min(x_coords)
                height = np.max(y_coords) - np.min(y_coords)
                aspect_ratio = width / (height + 1e-6)
                
                print(f"  Aspect ratio (W/H): {aspect_ratio:.2f}")
            
            # Predict voi window=15
            buffer = deque(maxlen=15)
            start_frame = max(0, target_frame - 15)
            
            for i in range(start_frame, target_frame + 1):
                if all_skeletons_norm[i] is not None:
                    buffer.append(all_skeletons_norm[i])
            
            if len(buffer) >= 15:
                prob = self.predict(buffer)
                print(f"\n  Fall Probability (window=15): {prob*100:.1f}%")
                status = "FALL" if prob >= 0.4 else "NORMAL"
                print(f"  Status: {status}")
            else:
                prob = 0.0
                print(f"\n  ⚠ Not enough frames for prediction ({len(buffer)}/15)")
            
            results[time_label] = {
                'detected': True,
                'skeleton_raw': skeleton_raw,
                'skeleton_norm': skeleton_norm,
                'probability': prob,
                'body_angle': angle if skeleton_norm is not None else 0,
                'aspect_ratio': aspect_ratio if skeleton_norm is not None else 0
            }
        
        # So sanh
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Time':<10} {'Detected':<10} {'Prob':<10} {'Angle':<10} {'Aspect':<10} {'Status':<10}")
        print("-"*70)
        
        for time_label, result in results.items():
            detected = "✓" if result['detected'] else "✗"
            prob = f"{result['probability']*100:.1f}%"
            angle = f"{result.get('body_angle', 0):.1f}°"
            aspect = f"{result.get('aspect_ratio', 0):.2f}"
            status = "FALL" if result['probability'] >= 0.4 else "NORMAL"
            
            print(f"{time_label:<10} {detected:<10} {prob:<10} {angle:<10} {aspect:<10} {status:<10}")
        
        # Ve bieu do
        print("\n" + "="*70)
        print("VISUALIZING SKELETONS")
        print("="*70)
        
        fig = plt.figure(figsize=(15, 5))
        
        for idx, (time_label, result) in enumerate(results.items()):
            if not result['detected'] or result['skeleton_norm'] is None:
                continue
            
            ax = fig.add_subplot(1, len(results), idx+1)
            
            skeleton = result['skeleton_norm']
            
            # Ve keypoints
            for i in range(17):
                if skeleton[i, 2] > 0.3:
                    ax.scatter(skeleton[i, 0], -skeleton[i, 1], c='red', s=50, zorder=5)
                    ax.annotate(str(i), (skeleton[i, 0], -skeleton[i, 1]), fontsize=8)
            
            # Ve connections
            connections = [
                (5, 6), (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            
            for start, end in connections:
                if skeleton[start, 2] > 0.3 and skeleton[end, 2] > 0.3:
                    ax.plot([skeleton[start, 0], skeleton[end, 0]],
                           [-skeleton[start, 1], -skeleton[end, 1]],
                           'b-', linewidth=2)
            
            ax.set_title(f"{time_label}\nProb: {result['probability']*100:.1f}%\nAngle: {result.get('body_angle', 0):.1f}°")
            ax.set_xlabel('X (normalized)')
            ax.set_ylabel('Y (normalized)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
        
        plt.tight_layout()
        plt.savefig('skeleton_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: skeleton_comparison.png")
        
        # Ket luan
        print("\n" + "="*70)
        print("CONCLUSIONS")
        print("="*70)
        
        # So sanh giay 5 va giay 11
        if '5.0s' in results and '11.0s' in results:
            r5 = results['5.0s']
            r11 = results['11.0s']
            
            print("\nComparing 5s (MISS) vs 11s (HIT):")
            
            if r5['detected'] and r11['detected']:
                print(f"  Body angle: {r5.get('body_angle', 0):.1f}° vs {r11.get('body_angle', 0):.1f}°")
                print(f"  Aspect ratio: {r5.get('aspect_ratio', 0):.2f} vs {r11.get('aspect_ratio', 0):.2f}")
                print(f"  Probability: {r5['probability']*100:.1f}% vs {r11['probability']*100:.1f}%")
                
                print("\nPossible reasons for MISS at 5s:")
                if r5.get('body_angle', 0) < 45:
                    print("  - Body angle too small (< 45°) - person not tilted enough")
                if r5.get('aspect_ratio', 0) < 1.2:
                    print("  - Aspect ratio too small (< 1.2) - person not horizontal")
                if r5['probability'] < 0.3:
                    print("  - Model confidence too low")
                print("  - Pose might be different from training data")
                print("  - Need more diverse training data")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze skeleton at specific times')
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to video')
    
    args = parser.parse_args()
    
    analyzer = SkeletonAnalyzer()
    analyzer.analyze_video(args.video, target_times=[1.0, 5.0, 11.0])

if __name__ == '__main__':
    main()
