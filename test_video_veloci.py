"""
Script test model TCN (có Velocity) tren video thuc te
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

# Import model TCN co Velocity
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_tcn_veloci import FallTCN, add_velocity_features

# Cau hinh
CONFIG = {
    'model_path': 'models/skeleton_veloci/TCN_Veloci_best.pth',
    'yolo_model': 'yolov8m-pose.pt',
    'window_size': 30,
    'num_features': 102, # Da tang len 102 (51 spatial + 51 velocity)
    'threshold': 0.5,  # TANG NGUONG LEN 0.85 (Cu la 0.5) DE LOAI BO HAN FALSE ALARM LOP SITTING DOWN
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Thu muc luu output
OUTPUT_DIR = 'test_outputs_veloci'
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
        print("  FALL DETECTION - VIDEO TESTER (TCN VELOCITY MODEL)")
        print("="*70)
        print(f"Device: {CONFIG['device']}")
        if CONFIG['device'] == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load TCN model
        print("\n[1/2] Loading TCN Velocity model...")
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
        print("      ✓ TCN Velocity model loaded")
        
        # Load YOLOv8-pose
        print("[2/2] Loading YOLOv8-pose model...")
        self.yolo = YOLO(CONFIG['yolo_model'])
        print("      ✓ YOLOv8-pose model loaded")
        
        # Buffer de luu 30 frames gan nhat cho tung nguoi (theo ID)
        self.skeleton_buffers = {}
        
        # Ket qua cho tung nguoi
        self.fall_probabilities = {}
        self.statuses = {}
        
        # Tracking nguoi qua cac frame
        self.next_id = 0
        self.tracked_bboxes = {}
        self.iou_threshold = 0.3
        
        # Timer hien thi canh bao FALL (so frame)
        self.fall_alert_timers = {}
        self.fall_alert_duration = 60
        
        # BO LOC: Dem so frame lien tiep vuot nguong de chong nhieu
        self.fall_consecutive_frames = {}
        
        print("\n" + "="*70)
        print("  SAN SANG TEST VIDEO")
        print("="*70)
    
    def detect_person(self, frame):
        """Phat hien nguoi va keypoints bang YOLOv8-pose"""
        results = self.yolo(frame, conf=0.3, verbose=False)
        
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
    
    def compute_iou(self, box1, box2):
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
        matched = {}
        new_tracked = {}
        used_det = set()
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
        for i, det in enumerate(detections):
            if i not in used_det:
                new_tracked[self.next_id] = det[:4]
                matched[self.next_id] = i
                self.next_id += 1
        self.tracked_bboxes = new_tracked
        return matched
    
    def normalize_skeleton(self, skeleton):
        """Chuan hoa skeleton giong nhu khi train"""
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
        if len(buffer) < CONFIG['window_size']:
            return 0.0
        
        window = np.array(buffer)
        window_flat = window.reshape(window.shape[0], -1)
        
        input_tensor = torch.FloatTensor(window_flat).unsqueeze(0)
        input_tensor = input_tensor.to(CONFIG['device'])
        
        # TÍNH TOÁN VELOCITY NGAY TRONG LÚC INFERENCE
        input_tensor = add_velocity_features(input_tensor)
        
        with torch.no_grad():
            prob = self.model(input_tensor).item()
        
        return prob
    
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        
        detections, keypoints_list = self.detect_person(frame)
        
        if detections:
            matched = self.track_persons(detections)
            
            active_ids = set(matched.keys())
            for pid in list(self.statuses.keys()):
                if pid not in active_ids:
                    del self.statuses[pid]
                    del self.fall_probabilities[pid]
                    del self.skeleton_buffers[pid]
                    if pid in self.fall_alert_timers:
                        del self.fall_alert_timers[pid]
                    if pid in self.fall_consecutive_frames:
                        del self.fall_consecutive_frames[pid]
            
            for person_id, det_idx in matched.items():
                bbox = detections[det_idx]
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                
                if person_id not in self.skeleton_buffers:
                    self.skeleton_buffers[person_id] = deque(maxlen=CONFIG['window_size'])
                    self.fall_probabilities[person_id] = 0.0
                    self.statuses[person_id] = "NORMAL"
                    self.fall_alert_timers[person_id] = 0
                    self.fall_consecutive_frames[person_id] = 0
                
                buffer = self.skeleton_buffers[person_id]
                
                if self.fall_alert_timers.get(person_id, 0) > 0:
                    self.fall_alert_timers[person_id] -= 1
                
                status = self.statuses.get(person_id, "NORMAL")
                box_color = (0, 0, 255) if status == "FALL" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                skeleton = keypoints_list.get(det_idx)
                
                if skeleton is not None:
                    self.draw_skeleton(frame, skeleton, 0, 0)
                    skeleton_norm = self.normalize_skeleton(skeleton)
                    
                    if skeleton_norm is not None:
                        buffer.append(skeleton_norm)
                        
                        if len(buffer) >= CONFIG['window_size']:
                            prob = self.predict(buffer)
                            self.fall_probabilities[person_id] = prob
                            
                            # BO LOC CHONG NHIEU SITTING DOWN
                            if prob >= CONFIG['threshold']:
                                self.fall_consecutive_frames[person_id] += 1
                            else:
                                self.fall_consecutive_frames[person_id] = 0
                                
                            # Chi bao NGA neu vuot nguong lien tuc 5 frames (~0.15s)
                            if self.fall_consecutive_frames[person_id] >= 5:
                                if self.statuses.get(person_id) != "FALL":
                                    import os, time
                                    save_dir = "alerts"
                                    os.makedirs(save_dir, exist_ok=True)
                                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                                    filename = os.path.join(save_dir, f"fall_detected_p{person_id}_{timestamp}.jpg")
                                    import cv2
                                    cv2.imwrite(filename, frame)
                                    print(f"  [ALERT] Phat hien NGA! Da luu anh: {filename}")
                                    
                                self.statuses[person_id] = "FALL"
                                self.fall_alert_timers[person_id] = self.fall_alert_duration
                            elif self.fall_alert_timers.get(person_id, 0) == 0:
                                self.statuses[person_id] = "NORMAL"
        
        self.draw_results(frame)
        return frame
    
    def draw_skeleton(self, frame, skeleton, offset_x, offset_y):
        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        for idx in range(min(17, len(skeleton))):
            x, y, conf = skeleton[idx]
            if conf > 0.3:
                px = int(x + offset_x)
                py = int(y + offset_y)
                cv2.circle(frame, (px, py), 3, (255, 200, 100), -1)
        
        for start, end in connections:
            if start < len(skeleton) and end < len(skeleton):
                if skeleton[start, 2] > 0.3 and skeleton[end, 2] > 0.3:
                    p1 = (int(skeleton[start, 0] + offset_x), int(skeleton[start, 1] + offset_y))
                    p2 = (int(skeleton[end, 0] + offset_x), int(skeleton[end, 1] + offset_y))
                    cv2.line(frame, p1, p2, (255, 200, 100), 2)
    
    def draw_results(self, frame):
        h, w = frame.shape[:2]
        any_fall = any(s == "FALL" for s in self.statuses.values())
        
        if any_fall:
            cv2.rectangle(frame, (5, 5), (w-5, h-5), (0, 0, 255), 5)
            
            alert_text = "FALL DETECTED!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            cv2.rectangle(frame, (text_x-10, text_y-40), (text_x+text_size[0]+10, text_y+10), (0, 0, 255), -1)
            cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    def test_video(self, video_path, output_path=None):
        print(f"\nOpening video: {video_path}")
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(OUTPUT_DIR, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f}s")
        
        out = None
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(video_output_dir, f"output_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"\nSaving output video to: {output_path}")
        
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
                
                processed_frame = self.process_frame(frame)
                cv2.imshow("Fall Detection Test", processed_frame)
                
                if out:
                    out.write(processed_frame)
                
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    status_info = ", ".join([f"ID:{pid}={s}" for pid, s in self.statuses.items()])
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {status_info}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuit by user")
                    break
                elif key == ord('s'):
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
            for pid in sorted(self.statuses.keys()):
                print(f"  ID:{pid} - Status: {self.statuses[pid]} - Prob: {self.fall_probabilities.get(pid, 0.0)*100:.1f}%")
            print(f"\nAll outputs saved in: {video_output_dir}")
            print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test TCN VELOCITY model on video')
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
