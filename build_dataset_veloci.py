import os, sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 30
STEP = 10
LEFT_SHOULDER = 5; RIGHT_SHOULDER = 6
LEFT_HIP = 11; RIGHT_HIP = 12
MIN_CONF = 0.3

DATA_DIR = r'C:\Download\Do_an\data_veloci'
OUTPUT_DIR = 'datasets/skeleton_windows_veloci'

def normalize_skeleton(skeleton):
    if skeleton is None: return None
    skeleton = skeleton.copy()
    xy = skeleton[:, :2]; conf = skeleton[:, 2]
    lh = xy[LEFT_HIP]; rh = xy[RIGHT_HIP]
    lhc = conf[LEFT_HIP]; rhc = conf[RIGHT_HIP]
    if lhc < MIN_CONF and rhc < MIN_CONF: return None
    if lhc >= MIN_CONF and rhc >= MIN_CONF: hc = (lh + rh) / 2
    elif lhc >= MIN_CONF: hc = lh
    else: hc = rh
    ls = xy[LEFT_SHOULDER]; rs = xy[RIGHT_SHOULDER]
    lsc = conf[LEFT_SHOULDER]; rsc = conf[RIGHT_SHOULDER]
    if lsc >= MIN_CONF and rsc >= MIN_CONF: sc = (ls + rs) / 2
    elif lsc >= MIN_CONF: sc = ls
    elif rsc >= MIN_CONF: sc = rs
    else: sc = hc
    bs = np.linalg.norm(sc - hc)
    if bs < 1e-6: bs = 1.0
    xy_n = (xy - hc) / bs
    xy_n[conf < MIN_CONF] = 0
    skeleton[:, :2] = xy_n
    return skeleton

def extract_windows_from_video(vpath, yolo):
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return []
    frames_data = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = yolo(frame, conf=0.5, verbose=False)[0]
        if results.keypoints is None or results.boxes is None:
            continue
        if len(results.boxes) == 0:
            continue
        keypoints_xy = results.keypoints.xy.cpu().numpy()
        keypoints_conf = results.keypoints.conf.cpu().numpy()
        bboxes = results.boxes.xyxy.cpu().numpy()
        best_idx = 0
        best_area = 0
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_idx = i
        skel = np.zeros((17, 3), dtype=np.float32)
        for j in range(17):
            skel[j] = [keypoints_xy[best_idx, j, 0], keypoints_xy[best_idx, j, 1], keypoints_conf[best_idx, j]]
        norm = normalize_skeleton(skel)
        if norm is not None:
            frames_data.append(norm.reshape(-1))
    cap.release()
    if len(frames_data) < WINDOW_SIZE:
        return []
    windows = []
    for start in range(0, len(frames_data) - WINDOW_SIZE + 1, STEP):
        window = np.array(frames_data[start:start + WINDOW_SIZE])
        windows.append(window)
    return windows

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    yolo = YOLO('yolov8m-pose.pt')
    yolo.to(DEVICE)
    X_all, y_all = [], []
    
    for cls_name in ['Fall', 'No_Fall']:
        cls_path = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_path):
            print(f"Warning: {cls_path} not found.")
            continue
        label = 1 if cls_name == 'Fall' else 0
        videos = sorted([f for f in os.listdir(cls_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))])
        for vid in videos:
            vpath = os.path.join(cls_path, vid)
            print(f"  Processing {cls_name}/{vid} ...", end=' ')
            sys.stdout.flush()
            windows = extract_windows_from_video(vpath, yolo)
            if windows:
                X_all.extend(windows)
                y_all.extend([label] * len(windows))
                print(f"{len(windows)} windows")
            else:
                print("SKIP (no poses or too short)")
                
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    
    if len(X) == 0:
        print("Error: No data extracted.")
        return
        
    print(f"\nExtracted total: {len(X)} windows.")
    
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    
    print(f"\nSaved to {OUTPUT_DIR}")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

if __name__ == '__main__':
    main()
