"""
Evaluate TCN + YOLOv8-pose on video folders (fall/ and No_fall/)
Headless, outputs metrics to terminal and JSON
"""
import os, sys, json, time, argparse
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        return super().default(obj)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_tcn_only import FallTCN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 30
THRESHOLD = 0.5
MOTION_THRESHOLD = 0.5  # Normalized motion threshold (fall >> 0.5, normal activity < 0.5)
LEFT_SHOULDER = 5; RIGHT_SHOULDER = 6
LEFT_HIP = 11; RIGHT_HIP = 12
MIN_CONF = 0.3

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

def evaluate_video(vpath, yolo, tcn_model):
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
                    if prob >= THRESHOLD and motion >= MOTION_THRESHOLD and fall_frame == -1:
                        fall_frame = frame_count
            break

    cap.release()
    has_motion = max_motion >= MOTION_THRESHOLD
    final_pred = 1 if (max_prob >= THRESHOLD and has_motion) else 0
    return {
        'video': os.path.basename(vpath),
        'total_frames': total_frames,
        'fps': fps,
        'frames_with_pose': frames_with_pose,
        'max_probability': round(float(max_prob), 4),
        'max_motion': round(float(max_motion), 4),
        'prediction': 'Fall' if final_pred else 'No_Fall',
        'fall_frame': fall_frame
    }

def main():
    global THRESHOLD, MOTION_THRESHOLD
    parser = argparse.ArgumentParser(description='Evaluate TCN+YOLOv8-pose')
    parser.add_argument('--base', default=r'C:\Download\Do_an\videoevaluate2',
                        help='Base dir with fall/ and No_fall/ subdirs')
    parser.add_argument('--threshold', '-t', type=float, default=0.75)
    parser.add_argument('--motion', '-m', type=float, default=0.5,
                        help='Motion threshold (default: 0.5)')
    parser.add_argument('--output', '-o', default='evaluation_results.json')
    args = parser.parse_args()

    THRESHOLD = args.threshold
    MOTION_THRESHOLD = args.motion

    print("="*70)
    print("  EVALUATE: TCN + YOLOv8-POSE")
    print("="*70)
    print(f"Device: {DEVICE}, Threshold: {THRESHOLD}")

    print("\n[1/2] Loading models...")
    tcn = FallTCN(51, [64, 128, 128], 3, 0.3)
    tcn.load_state_dict(torch.load('models/skeleton/TCN_best.pth',
                                   map_location=DEVICE, weights_only=False)['model_state_dict'])
    tcn.to(DEVICE); tcn.eval()
    print("  ✓ TCN loaded")

    yolo = YOLO('yolov8m-pose.pt')
    print("  ✓ YOLOv8-pose loaded")

    fall_dir = os.path.join(args.base, 'fall')
    nofall_dir = os.path.join(args.base, 'No_fall')

    videos = []
    for f in sorted(os.listdir(fall_dir)):
        if f.lower().endswith('.mp4'):
            videos.append(('Fall', os.path.join(fall_dir, f)))
    for f in sorted(os.listdir(nofall_dir)):
        if f.lower().endswith('.mp4'):
            videos.append(('No_Fall', os.path.join(nofall_dir, f)))

    print(f"\n[2/2] Evaluating {len(videos)} videos "
          f"({sum(1 for v in videos if v[0]=='Fall')}F / "
          f"{sum(1 for v in videos if v[0]=='No_Fall')}NF)...")

    results = []
    t_start = time.time()
    for idx, (gt, vpath) in enumerate(videos):
        t0 = time.time()
        vname = os.path.basename(vpath)
        res = evaluate_video(vpath, yolo, tcn)
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
        print(f"  [{idx+1:2d}/{len(videos)}] {vname:20s} GT={gt:8s} -> {pred:8s} (prob={prob:.3f} mot={mot:.2f}) "
              f"[{elapsed:.0f}s, ETA={etc:.0f}s]")

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  RESULTS — {len(results)} videos ({total_time:.0f}s)")
    print(f"{'='*70}")

    tp = sum(1 for r in results if r['ground_truth']=='Fall' and r.get('prediction')=='Fall')
    fp = sum(1 for r in results if r['ground_truth']=='No_Fall' and r.get('prediction')=='Fall')
    tn = sum(1 for r in results if r['ground_truth']=='No_Fall' and r.get('prediction')=='No_Fall')
    fn = sum(1 for r in results if r['ground_truth']=='Fall' and r.get('prediction')=='No_Fall')
    total = tp+fp+tn+fn
    acc = (tp+tn)/total if total else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0

    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Accuracy:     {acc*100:.2f}%")
    print(f"  Precision:    {prec*100:.2f}%")
    print(f"  Recall:       {rec*100:.2f}%")
    print(f"  Specificity:  {spec*100:.2f}%")
    print(f"  F1-Score:     {f1*100:.2f}%")
    print(f"  False Alarm:  {(fp/(fp+tn)*100) if (fp+tn) else 0:.2f}%")
    print(f"  Miss Rate:    {(fn/(tp+fn)*100) if (tp+fn) else 0:.2f}%")

    output = {
        'config': {'device': DEVICE, 'threshold': THRESHOLD, 'window_size': WINDOW_SIZE},
        'dataset': {'path': args.base, 'total_videos': len(videos),
                     'fall': sum(1 for v in videos if v[0]=='Fall'),
                     'no_fall': sum(1 for v in videos if v[0]=='No_Fall')},
        'metrics': {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'accuracy': round(acc*100, 2), 'precision': round(prec*100, 2),
            'recall': round(rec*100, 2), 'specificity': round(spec*100, 2),
            'f1': round(f1*100, 2),
            'false_alarm_rate': round(fp/(fp+tn)*100, 2) if (fp+tn) else 0,
            'miss_rate': round(fn/(tp+fn)*100, 2) if (tp+fn) else 0
        },
        'details': results
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {args.output}")
    print("="*70)

if __name__ == '__main__':
    main()
