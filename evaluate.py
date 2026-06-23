"""
Unified evaluator: TCN+YOLOv8-pose, LSTM, threshold tuning, subject4, rule-based, all-methods.

Usage:
  python evaluate.py --mode tcn --base path/to/videos
  python evaluate.py --mode lstm
  python evaluate.py --mode thresholds [--input results.json]
  python evaluate.py --mode subject4 --base "path/to/Subject 4"
  python evaluate.py --mode rule-based --dataset path/to/dataset
  python evaluate.py --mode all --base path/to/videos
"""
import os, sys, json, time, argparse, csv
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_tcn_only import FallTCN
import config as cfg
from main import FallDetectionSystem

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 30
MIN_CONF = 0.3
LEFT_SHOULDER = 5; RIGHT_SHOULDER = 6; LEFT_HIP = 11; RIGHT_HIP = 12

# ============================================================
#  SHARED UTILITIES
# ============================================================

class FallLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        return super().default(obj)

def normalize_skeleton(skeleton):
    if skeleton is None: return None
    skeleton = skeleton.copy()
    xy = skeleton[:, :2]; conf = skeleton[:, 2]
    lh, rh = xy[LEFT_HIP], xy[RIGHT_HIP]
    lhc, rhc = conf[LEFT_HIP], conf[RIGHT_HIP]
    if lhc < MIN_CONF and rhc < MIN_CONF: return None
    if lhc >= MIN_CONF and rhc >= MIN_CONF: hc = (lh + rh) / 2
    elif lhc >= MIN_CONF: hc = lh
    else: hc = rh
    ls, rs = xy[LEFT_SHOULDER], xy[RIGHT_SHOULDER]
    lsc, rsc = conf[LEFT_SHOULDER], conf[RIGHT_SHOULDER]
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

def load_tcn(path='models/skeleton/TCN_best.pth'):
    model = FallTCN(51, [64, 128, 128], 3, 0.3)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE); model.eval()
    return model

def load_lstm(path='models/skeleton/LSTM_best.pth'):
    model = FallLSTM(51, 128, 2, 0.3)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE); model.eval()
    return model

def print_metrics(tp, fp, tn, fn, label=''):
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    far = fp / (fp + tn) if (fp + tn) else 0
    mr = fn / (tp + fn) if (tp + fn) else 0
    print(f"\n{label}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Accuracy:     {acc*100:.2f}%")
    print(f"  Precision:    {prec*100:.2f}%")
    print(f"  Recall:       {rec*100:.2f}%")
    print(f"  Specificity:  {spec*100:.2f}%")
    print(f"  F1-Score:     {f1*100:.2f}%")
    print(f"  False Alarm:  {far*100:.2f}%")
    print(f"  Miss Rate:    {mr*100:.2f}%")
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'accuracy': round(acc*100, 2), 'precision': round(prec*100, 2),
            'recall': round(rec*100, 2), 'specificity': round(spec*100, 2),
            'f1': round(f1*100, 2), 'false_alarm_rate': round(far*100, 2),
            'miss_rate': round(mr*100, 2)}

# ============================================================
#  MODE: TCN (on video folders with YOLOv8-pose)
# ============================================================

def evaluate_video_tcn(vpath, yolo, tcn_model, threshold=0.75, motion_threshold=0.5):
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return None
    buffer = deque(maxlen=WINDOW_SIZE)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    max_prob = 0.0
    max_motion = 0.0
    fall_frame = -1
    frames_with_pose = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        results = yolo(frame, conf=0.5, verbose=False)[0]
        if results.keypoints is None or results.boxes is None:
            continue
        if len(results.boxes) == 0:
            continue

        keypoints_xy = results.keypoints.xy.cpu().numpy()
        keypoints_conf = results.keypoints.conf.cpu().numpy()

        for i in range(len(results.boxes)):
            skeleton = np.zeros((17, 3), dtype=np.float32)
            skeleton[:, :2] = keypoints_xy[i]
            skeleton[:, 2] = keypoints_conf[i]

            sk_n = normalize_skeleton(skeleton)
            if sk_n is not None:
                frames_with_pose += 1
                buffer.append(sk_n)
                if len(buffer) >= WINDOW_SIZE:
                    window = np.array(buffer).reshape(1, WINDOW_SIZE, -1)
                    inp = torch.FloatTensor(window).to(DEVICE)
                    with torch.no_grad():
                        prob = tcn_model(inp).item()

                    vector = np.array(list(buffer))
                    motion = float(np.max(np.linalg.norm(vector[1:] - vector[:-1], axis=2).mean(axis=1)))

                    if motion > max_motion:
                        max_motion = motion
                    if prob > max_prob:
                        max_prob = prob
                    if prob >= threshold and motion >= motion_threshold and fall_frame == -1:
                        fall_frame = frame_count
            break

    cap.release()
    has_motion = max_motion >= motion_threshold
    final_pred = 'Fall' if (max_prob >= threshold and has_motion) else 'No_Fall'
    return {
        'video': os.path.basename(vpath),
        'total_frames': total_frames,
        'fps': fps,
        'frames_with_pose': frames_with_pose,
        'max_probability': round(float(max_prob), 4),
        'max_motion': round(float(max_motion), 4),
        'prediction': final_pred,
        'fall_frame': fall_frame
    }

def run_tcn(base, threshold=0.75, motion=0.5, output='evaluation_results.json'):
    print(f"\n  MODE: TCN+YOLOv8-pose (threshold={threshold}, motion={motion})")
    print(f"  Base: {base}")
    tcn = load_tcn()
    yolo = YOLO('yolov8m-pose.pt')

    fall_dir = os.path.join(base, 'fall')
    nofall_dir = os.path.join(base, 'No_fall')
    videos = []
    for f in sorted(os.listdir(fall_dir)):
        if f.lower().endswith('.mp4'):
            videos.append(('Fall', os.path.join(fall_dir, f)))
    for f in sorted(os.listdir(nofall_dir)):
        if f.lower().endswith('.mp4'):
            videos.append(('No_Fall', os.path.join(nofall_dir, f)))

    print(f"  Videos: {len(videos)} ({sum(1 for v in videos if v[0]=='Fall')}F / "
          f"{sum(1 for v in videos if v[0]=='No_Fall')}NF)")

    results = []
    t_start = time.time()
    for idx, (gt, vpath) in enumerate(videos):
        t0 = time.time()
        vname = os.path.basename(vpath)
        res = evaluate_video_tcn(vpath, yolo, tcn, threshold, motion)
        if res:
            res['ground_truth'] = gt
            results.append(res)
            elapsed = time.time() - t0
        else:
            elapsed = time.time() - t0
            res = {'video': vname, 'prediction': 'FAILED', 'ground_truth': gt}
            results.append(res)

        etc = (time.time() - t_start) / (idx + 1) * (len(videos) - idx - 1)
        pred = res.get('prediction', 'FAILED')[:8]
        prob = res.get('max_probability', 0)
        mot = res.get('max_motion', 0)
        print(f"  [{idx+1:2d}/{len(videos)}] {vname:20s} GT={gt:8s} -> {pred:8s} "
              f"(prob={prob:.3f} mot={mot:.2f}) [{elapsed:.0f}s, ETA={etc:.0f}s]")

    tp = sum(1 for r in results if r['ground_truth']=='Fall' and r.get('prediction')=='Fall')
    fp = sum(1 for r in results if r['ground_truth']=='No_Fall' and r.get('prediction')=='Fall')
    tn = sum(1 for r in results if r['ground_truth']=='No_Fall' and r.get('prediction')=='No_Fall')
    fn = sum(1 for r in results if r['ground_truth']=='Fall' and r.get('prediction')=='No_Fall')
    m = print_metrics(tp, fp, tn, fn, "  RESULTS")

    output_data = {
        'config': {'device': DEVICE, 'threshold': threshold, 'motion_threshold': motion,
                    'window_size': WINDOW_SIZE, 'model': 'TCN+YOLOv8-pose'},
        'dataset': {'path': base, 'total_videos': len(videos),
                     'fall': sum(1 for v in videos if v[0]=='Fall'),
                     'no_fall': sum(1 for v in videos if v[0]=='No_Fall')},
        'metrics': m,
        'details': results
    }
    with open(output, 'w') as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved to {output}")

# ============================================================
#  MODE: LSTM (on .npy test set)
# ============================================================

def run_lstm():
    print("\n  MODE: LSTM on test set")
    data_dir = 'datasets/skeleton_windows'
    model_dir = 'models/skeleton'

    X_test = torch.FloatTensor(np.load(os.path.join(data_dir, 'X_test.npy')))
    y_test = torch.FloatTensor(np.load(os.path.join(data_dir, 'y_test.npy')))

    model = load_lstm(os.path.join(model_dir, 'LSTM_best.pth'))
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            out = model(Xb.to(DEVICE))
            preds = (out >= 0.5).float().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(yb.numpy())

    all_preds = np.array(all_preds); all_labels = np.array(all_labels)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    print_metrics(tp, fp, tn, fn, "  LSTM RESULTS")
    print(f"\n  {classification_report(all_labels, all_preds, target_names=['Normal', 'Fall'])}")

# ============================================================
#  MODE: THRESHOLDS (analyze from existing JSON)
# ============================================================

def run_thresholds(input_path='evaluation_results.json'):
    print(f"\n  MODE: Threshold analysis from {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    details = data.get('details', data)

    # Detect available methods
    methods = []
    for key in ['tcn_yolov8n_prob', 'tcn_yolov8m_prob', 'max_probability']:
        if any(key in r for r in details):
            mname = key.replace('_prob', '').replace('_probability', '')
            methods.append((mname, key))

    for mname, pk in methods:
        print(f"\n=== {mname.upper()} ===")
        print(f"{'Thresh':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'Spec':<8} {'F1':<8} "
              f"{'TP':<4} {'FP':<4} {'TN':<4} {'FN':<4}")
        for thresh in [x/100 for x in range(50, 96, 5)]:
            tp = sum(1 for r in details if r['ground_truth']=='Fall' and r.get(pk, 0) >= thresh)
            fp = sum(1 for r in details if r['ground_truth']=='No_Fall' and r.get(pk, 0) >= thresh)
            tn = sum(1 for r in details if r['ground_truth']=='No_Fall' and r.get(pk, 0) < thresh)
            fn = sum(1 for r in details if r['ground_truth']=='Fall' and r.get(pk, 0) < thresh)
            total = tp+fp+tn+fn
            acc = (tp+tn)/total*100 if total else 0
            prec = tp/(tp+fp)*100 if (tp+fp) else 0
            rec = tp/(tp+fn)*100 if (tp+fn) else 0
            spec = tn/(tn+fp)*100 if (tn+fp) else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
            print(f'{thresh:<8.2f} {acc:<8.2f} {prec:<8.2f} {rec:<8.2f} {spec:<8.2f} {f1:<8.2f} '
                  f'{tp:<4} {fp:<4} {tn:<4} {fn:<4}')

# ============================================================
#  MODE: SUBJECT4 (pre vs post fine-tune)
# ============================================================

def compute_motion(buffer):
    if len(buffer) < 2: return 0.0
    arr = np.array(list(buffer))
    motion = np.mean(np.abs(np.diff(arr.reshape(arr.shape[0], -1), axis=0)))
    return motion

def eval_model_on_subject(model, yolo, base_path, threshold=0.75, motion_threshold=0.25):
    tp = fp = tn = fn = 0
    results = []
    for cls_name, label in [('Fall', 1), ('No_Fall', 0)]:
        cls_path = os.path.join(base_path, cls_name)
        if not os.path.isdir(cls_path): continue
        for mp4 in sorted([f for f in os.listdir(cls_path) if f.endswith('.mp4')]):
            vpath = os.path.join(cls_path, mp4)
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened(): continue
            buffer = deque(maxlen=WINDOW_SIZE)
            max_prob = 0.0; max_motion = 0.0; predicted = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                r = yolo(frame, conf=0.5, verbose=False)[0]
                if r.keypoints is None or r.boxes is None: continue
                if len(r.boxes) == 0: continue
                kp_xy = r.keypoints.xy.cpu().numpy()
                kp_c = r.keypoints.conf.cpu().numpy()
                bboxes = r.boxes.xyxy.cpu().numpy()
                best = max(range(len(bboxes)), key=lambda i: (bboxes[i][2]-bboxes[i][0])*(bboxes[i][3]-bboxes[i][1]))
                skel = np.zeros((17,3), dtype=np.float32)
                for j in range(17):
                    skel[j] = [kp_xy[best, j, 0], kp_xy[best, j, 1], kp_c[best, j]]
                norm = normalize_skeleton(skel)
                if norm is not None:
                    buffer.append(norm)
                    motion = compute_motion(buffer)
                    if motion > max_motion: max_motion = motion
                    if len(buffer) >= WINDOW_SIZE:
                        inp = torch.FloatTensor(np.array(buffer).reshape(1, WINDOW_SIZE, 51)).to(DEVICE)
                        with torch.no_grad():
                            prob = model(inp).item()
                        if prob > max_prob: max_prob = prob
                        if motion >= motion_threshold and prob >= threshold:
                            predicted = 1
            cap.release()
            if label == 1:
                if predicted == 1: tp += 1
                else: fn += 1
            else:
                if predicted == 1: fp += 1
                else: tn += 1
            results.append({
                'video': f"{cls_name}/{mp4}", 'label': label, 'predicted': predicted,
                'max_prob': round(max_prob, 4), 'max_motion': round(max_motion, 4)
            })
    return print_metrics(tp, fp, tn, fn) | {'details': results}

def run_subject4(base_path):
    print(f"\n  MODE: Subject 4 comparison (base={base_path})")
    yolo = YOLO('yolov8m-pose.pt')

    for motion_t in [0, 0.25]:
        print(f"\n{'='*70}")
        print(f"  MOTION THRESHOLD = {motion_t}")
        print(f"{'='*70}")
        for model_name, model_path in [('BEFORE fine-tune', 'models/skeleton/TCN_best.pth'),
                                        ('AFTER fine-tune',  'models/skeleton/TCN_finetuned.pth')]:
            print(f"\n--- {model_name} ---")
            model = load_tcn(model_path)
            m = eval_model_on_subject(model, yolo, base_path, threshold=0.75, motion_threshold=motion_t)
            if motion_t == 0.25:
                fname = f'model_{model_name.replace(" ", "_").lower().replace("-","_")}.json'
                with open(fname, 'w') as f:
                    json.dump(m, f, indent=2, cls=NumpyEncoder)

# ============================================================
#  MODE: RULE-BASED (on dataset folders)
# ============================================================

def evaluate_video_rule(system, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    fall_detected = False
    system.fall_detector.reset()
    while True:
        ret, frame = cap.read()
        if not ret: break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        _, fall_info = system.process_frame(frame, timestamp=timestamp)
        if fall_info['status'] == 'FALL':
            fall_detected = True
            break
    cap.release()
    return fall_detected

def run_rule_based(dataset_dir, limit=0):
    print(f"\n  MODE: Rule-based (dataset={dataset_dir}, limit={limit})")
    cfg.ALERT_CONFIG['sound_enabled'] = False
    cfg.ALERT_CONFIG['save_fall_images'] = False

    fall_dir = os.path.join(dataset_dir, 'Fall', 'Raw_Video')
    nofall_dir = os.path.join(dataset_dir, 'No_Fall', 'Raw_Video')

    import glob
    fall_videos = glob.glob(os.path.join(fall_dir, '*.mp4')) + glob.glob(os.path.join(fall_dir, '*.avi'))
    nofall_videos = glob.glob(os.path.join(nofall_dir, '*.mp4')) + glob.glob(os.path.join(nofall_dir, '*.avi'))

    if limit > 0:
        import random
        random.seed(42)
        fall_videos = random.sample(fall_videos, min(limit, len(fall_videos)))
        nofall_videos = random.sample(nofall_videos, min(limit, len(nofall_videos)))

    print(f"  Fall videos: {len(fall_videos)}, No_Fall videos: {len(nofall_videos)}")
    system = FallDetectionSystem()
    TP = FN = TN = FP = 0

    for v in fall_videos:
        if evaluate_video_rule(system, v):
            TP += 1
        else:
            FN += 1

    for v in nofall_videos:
        if evaluate_video_rule(system, v):
            FP += 1
        else:
            TN += 1

    print_metrics(TP, FP, TN, FN, "  RULE-BASED RESULTS")

# ============================================================
#  MODE: ALL METHODS (compare on videoevaluate)
# ============================================================

from utils import FallDetector, PoseEstimator
import mediapipe as mp

FRAME_SKIP = 5

def extract_skeleton_mediapipe(frame, bbox, mp_pose):
    x1,y1,x2,y2 = [int(v) for v in bbox[:4]]
    h,w = frame.shape[:2]
    pad=20; x1c=max(0,x1-pad); y1c=max(0,y1-pad); x2c=min(w,x2+pad); y2c=min(h,y2+pad)
    region = frame[y1c:y2c, x1c:x2c]
    if region.size == 0: return None
    rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    res = mp_pose.process(rgb)
    if not res.pose_landmarks: return None
    skel = np.full((17,3), np.nan)
    for idx, lm in enumerate(res.pose_landmarks.landmark):
        if idx < 17:
            skel[idx] = [lm.x*(x2c-x1c), lm.y*(y2c-y1c), lm.visibility]
    return skel

def evaluate_video_all(vpath, yolo_n, yolo_m, pose_est, fall_det, mp_pose, tcn_model, lstm_model):
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return {'rule_based':'No_Fall','tcn_yolov8n':'No_Fall','tcn_yolov8n_prob':0,
                'lstm':'No_Fall','lstm_prob':0,'tcn_yolov8m':'No_Fall','tcn_yolov8m_prob':0}
    fall_det.reset()
    buf_n = deque(maxlen=WINDOW_SIZE)
    buf_m = deque(maxlen=WINDOW_SIZE)
    rb_result = 'No_Fall'
    max_prob_n = max_prob_m = max_prob_lstm = 0.0
    tcn_n_pred = tcn_m_pred = lstm_pred = 'No_Fall'
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

        results_n = yolo_n(frame, conf=0.5, classes=[0], verbose=False)
        dets_n = [[*box.xyxy[0].cpu().numpy(), box.conf[0].cpu().numpy()] for r in results_n for box in r.boxes]
        results_m = yolo_m(frame, conf=0.5, classes=[0], verbose=False)
        dets_m = [[*box.xyxy[0].cpu().numpy()] for r in results_m for box in r.boxes]

        # Rule-based
        if dets_n and rb_result == 'No_Fall':
            x1,y1,x2,y2 = [int(v) for v in dets_n[0][:4]]
            h,w = frame.shape[:2]
            x1c=max(0,x1-20); y1c=max(0,y1-20); x2c=min(w,x2+20); y2c=min(h,y2+20)
            region = frame[y1c:y2c, x1c:x2c]
            if region.size > 0:
                landmarks, bbox_norm, _ = pose_est.process_frame(region)
                if landmarks and bbox_norm:
                    finfo = fall_det.detect(landmarks, bbox_norm, x2c-x1c, y2c-y1c, timestamp=ts)
                    if finfo['status'] == 'FALL':
                        rb_result = 'Fall'

        # TCN/LSTM (YOLOv8n)
        if dets_n:
            skel = extract_skeleton_mediapipe(frame, dets_n[0], mp_pose)
            if skel is not None:
                sk_n = normalize_skeleton(skel)
                if sk_n is not None:
                    buf_n.append(sk_n)
                    if len(buf_n) >= WINDOW_SIZE:
                        win = np.array(buf_n).reshape(1, WINDOW_SIZE, -1)
                        inp = torch.FloatTensor(win).to(DEVICE)
                        with torch.no_grad():
                            p_tcn = tcn_model(inp).item()
                            p_lstm = lstm_model(inp).item()
                        if p_tcn > max_prob_n: max_prob_n = p_tcn
                        if p_tcn >= 0.5: tcn_n_pred = 'Fall'
                        if p_lstm > max_prob_lstm: max_prob_lstm = p_lstm
                        if p_lstm >= 0.5: lstm_pred = 'Fall'

        # TCN (YOLOv8m)
        if dets_m:
            skel = extract_skeleton_mediapipe(frame, dets_m[0], mp_pose)
            if skel is not None:
                sk_m = normalize_skeleton(skel)
                if sk_m is not None:
                    buf_m.append(sk_m)
                    if len(buf_m) >= WINDOW_SIZE:
                        win = np.array(buf_m).reshape(1, WINDOW_SIZE, -1)
                        inp = torch.FloatTensor(win).to(DEVICE)
                        with torch.no_grad():
                            p_m = tcn_model(inp).item()
                        if p_m > max_prob_m: max_prob_m = p_m
                        if p_m >= 0.5: tcn_m_pred = 'Fall'

    cap.release()
    return {
        'rule_based': rb_result,
        'tcn_yolov8n': tcn_n_pred, 'tcn_yolov8n_prob': round(max_prob_n, 4),
        'lstm': lstm_pred, 'lstm_prob': round(max_prob_lstm, 4),
        'tcn_yolov8m': tcn_m_pred, 'tcn_yolov8m_prob': round(max_prob_m, 4)
    }

def run_all(base_dir):
    print(f"\n  MODE: All methods comparison (base={base_dir})")
    tcn = load_tcn()
    lstm = load_lstm()
    yolo_n = YOLO('yolov8n.pt')
    yolo_m = YOLO('yolov8m.pt')
    pose_est = PoseEstimator()
    fall_det = FallDetector()
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    videos = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith('.avi'):
                label = 'Fall' if 'Fall' in root.split(os.sep) else 'No_Fall'
                videos.append((label, os.path.join(root, f)))
    videos.sort(key=lambda x: x[1])
    print(f"  Found {len(videos)} videos ({sum(1 for v in videos if v[0]=='Fall')}F/"
          f"{sum(1 for v in videos if v[0]=='No_Fall')}NF)")

    results = []
    t_start = time.time()
    for idx, (gt, vpath) in enumerate(videos):
        vname = os.path.basename(vpath)
        t0 = time.time()
        res = evaluate_video_all(vpath, yolo_n, yolo_m, pose_est, fall_det, mp_pose, tcn, lstm)
        res['video'] = vname; res['ground_truth'] = gt
        results.append(res)
        elapsed = time.time()-t0
        total_elapsed = time.time()-t_start
        etc = (total_elapsed/(idx+1))*(len(videos)-idx-1)
        print(f"  [{idx+1}/{len(videos)}] {vname[:25]:25s} GT={gt:8s} "
              f"RB={res['rule_based']:8s} TCNn={res['tcn_yolov8n']:8s} "
              f"LSTM={res['lstm']:8s} TCNm={res['tcn_yolov8m']:8s} "
              f"({elapsed:.0f}s, ETA={etc:.0f}s)")

    methods = ['rule_based', 'tcn_yolov8n', 'lstm', 'tcn_yolov8m']
    all_metrics = {}
    for m in methods:
        tp = sum(1 for r in results if r['ground_truth']=='Fall' and r[m]=='Fall')
        fp = sum(1 for r in results if r['ground_truth']=='No_Fall' and r[m]=='Fall')
        tn = sum(1 for r in results if r['ground_truth']=='No_Fall' and r[m]=='No_Fall')
        fn = sum(1 for r in results if r['ground_truth']=='Fall' and r[m]=='No_Fall')
        all_metrics[m] = print_metrics(tp, fp, tn, fn, f"  {m.upper()}")

    output = {'metrics': all_metrics, 'details': results}
    with open('evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved to evaluation_results.json")

# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Fall Detection Evaluator')
    parser.add_argument('--mode', required=True,
                        choices=['tcn', 'lstm', 'thresholds', 'subject4', 'rule-based', 'all'],
                        help='Evaluation mode')
    parser.add_argument('--base', default=None, help='Base directory with fall/No_fall subdirs')
    parser.add_argument('--dataset', default=None, help='Dataset directory (rule-based mode)')
    parser.add_argument('--input', default='evaluation_results.json', help='Input JSON (thresholds mode)')
    parser.add_argument('--output', '-o', default='evaluation_results.json', help='Output JSON path')
    parser.add_argument('--threshold', '-t', type=float, default=0.75, help='TCN threshold')
    parser.add_argument('--motion', '-m', type=float, default=0.5, help='Motion threshold')
    parser.add_argument('--limit', '-l', type=int, default=0, help='Limit videos (rule-based)')
    args = parser.parse_args()

    print("=" * 70)
    print("  FALL DETECTION EVALUATOR")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    if args.mode == 'tcn':
        if not args.base:
            print("ERROR: --base is required for tcn mode")
            return
        run_tcn(args.base, args.threshold, args.motion, args.output)

    elif args.mode == 'lstm':
        run_lstm()

    elif args.mode == 'thresholds':
        run_thresholds(args.input)

    elif args.mode == 'subject4':
        if not args.base:
            print("ERROR: --base is required for subject4 mode")
            return
        run_subject4(args.base)

    elif args.mode == 'rule-based':
        if not args.dataset:
            print("ERROR: --dataset is required for rule-based mode")
            return
        run_rule_based(args.dataset, args.limit)

    elif args.mode == 'all':
        if not args.base:
            print("ERROR: --base is required for all mode")
            return
        run_all(args.base)

if __name__ == '__main__':
    main()
