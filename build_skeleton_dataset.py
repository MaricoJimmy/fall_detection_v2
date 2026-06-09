"""
Script xay dung skeleton dataset tu file CSV keypoints
Doc CSV -> Normalize -> Sliding Windows -> Chia train/val/test -> Luu .npy

Cau truc CSV: Frame, Keypoint, X, Y, Confidence
17 keypoints COCO format moi frame
"""
import os
import glob
import csv
import numpy as np
import argparse
from collections import OrderedDict

# === THU TU KEYPOINTS TRONG CSV (COCO format, 17 diem) ===
# Thu tu nay co dinh trong moi file CSV
KEYPOINT_NAMES = [
    'Nose',           # 0
    'Left Eye',       # 1
    'Right Eye',      # 2
    'Left Ear',       # 3
    'Right Ear',      # 4
    'Left Shoulder',  # 5
    'Right Shoulder', # 6
    'Left Elbow',     # 7
    'Right Elbow',    # 8
    'Left Wrist',     # 9
    'Right Wrist',    # 10
    'Left Hip',       # 11
    'Right Hip',      # 12
    'Left Knee',      # 13
    'Right Knee',     # 14
    'Left Ankle',     # 15
    'Right Ankle',    # 16
]

# Map ten keypoint -> index
KEYPOINT_TO_IDX = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

# === CHI SO CAC DIEM QUAN TRONG DE NORMALIZE ===
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

# === CAU HINH MAC DINH ===
DEFAULT_WINDOW_SIZE = 30    # So frame moi window
DEFAULT_STRIDE = 10         # Buoc nhay giua cac window (stride < window_size => overlap)
DEFAULT_MIN_FRAMES = 10     # So frame toi thieu de tao window (video qua ngan thi bo qua)
DEFAULT_MIN_CONFIDENCE = 0.3  # Confidence toi thieu de coi keypoint la hop le


def parse_csv(csv_path):
    """
    Doc file CSV keypoints va tra ve numpy array

    Args:
        csv_path: Duong dan file CSV

    Returns:
        numpy array co shape (num_frames, 17, 3) voi 3 = [X, Y, Confidence]
        Tra ve None neu file loi hoac qua ngan
    """
    # Dung OrderedDict de luu giu thu tu frame
    frames_data = OrderedDict()

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                frame_num = int(row['Frame'])
                keypoint_name = row['Keypoint'].strip()
                x = float(row['X'])
                y = float(row['Y'])
                confidence = float(row['Confidence'])

                # Bo qua keypoint khong co trong danh sach
                if keypoint_name not in KEYPOINT_TO_IDX:
                    continue

                # Khoi tao frame neu chua co
                if frame_num not in frames_data:
                    # Khoi tao mang NaN cho tat ca 17 keypoints
                    frames_data[frame_num] = np.full((17, 3), np.nan)

                # Dien du lieu vao dung vi tri
                kp_idx = KEYPOINT_TO_IDX[keypoint_name]
                frames_data[frame_num][kp_idx] = [x, y, confidence]

    except Exception as e:
        print(f"  [LOI] Khong doc duoc {csv_path}: {e}")
        return None

    if len(frames_data) == 0:
        return None

    # Chuyen thanh numpy array, sap xep theo thu tu frame
    sorted_frames = sorted(frames_data.keys())
    data = np.array([frames_data[f] for f in sorted_frames])

    return data


def normalize_skeleton(data):
    """
    Chuan hoa skeleton de giam anh huong cua:
    - Vi tri nguoi trong anh (dich chuyen)
    - Khoang cach camera (kich thuoc)
    - Chieu cao/nguoi khac nhau

    Cach lam:
    1. Lay hip center lam goc toa do (tru tat ca diem cho hip center)
    2. Chia cho body size (khoang cach vai-hip) de chuan hoa kich thuoc
    3. Thay the diem co confidence thap bang 0

    Args:
        data: numpy array (num_frames, 17, 3) voi [X, Y, Confidence]

    Returns:
        numpy array (num_frames, 17, 3) da chuan hoa
    """
    num_frames = data.shape[0]
    normalized = data.copy()

    for i in range(num_frames):
        frame = normalized[i]  # (17, 3)
        xy = frame[:, :2]      # (17, 2) - X, Y
        conf = frame[:, 2]     # (17,) - Confidence

        # === Buoc 1: Tinh hip center lam goc ===
        left_hip = xy[LEFT_HIP]
        right_hip = xy[RIGHT_HIP]
        left_hip_conf = conf[LEFT_HIP]
        right_hip_conf = conf[RIGHT_HIP]

        # Neu ca 2 hip deu khong nhin thay (confidence qua thap), bo qua frame nay
        if left_hip_conf < DEFAULT_MIN_CONFIDENCE and right_hip_conf < DEFAULT_MIN_CONFIDENCE:
            # Dat tat ca ve 0 (frame khong hop le)
            normalized[i, :, :2] = 0
            continue

        # Tinh hip center (trung binh 2 hip, hoac lay hip nao co confidence cao hon)
        if left_hip_conf >= DEFAULT_MIN_CONFIDENCE and right_hip_conf >= DEFAULT_MIN_CONFIDENCE:
            hip_center = (left_hip + right_hip) / 2
        elif left_hip_conf >= DEFAULT_MIN_CONFIDENCE:
            hip_center = left_hip
        else:
            hip_center = right_hip

        # === Buoc 2: Tinh body size (khoang cach vai-hip) ===
        left_shoulder = xy[LEFT_SHOULDER]
        right_shoulder = xy[RIGHT_SHOULDER]
        left_shoulder_conf = conf[LEFT_SHOULDER]
        right_shoulder_conf = conf[RIGHT_SHOULDER]

        # Tinh shoulder center tuong tu
        if left_shoulder_conf >= DEFAULT_MIN_CONFIDENCE and right_shoulder_conf >= DEFAULT_MIN_CONFIDENCE:
            shoulder_center = (left_shoulder + right_shoulder) / 2
        elif left_shoulder_conf >= DEFAULT_MIN_CONFIDENCE:
            shoulder_center = left_shoulder
        elif right_shoulder_conf >= DEFAULT_MIN_CONFIDENCE:
            shoulder_center = right_shoulder
        else:
            shoulder_center = hip_center  # Fallback

        # Body size = khoang cach giua shoulder center va hip center
        body_size = np.linalg.norm(shoulder_center - hip_center)
        if body_size < 1e-6:
            body_size = 1.0  # Tranh chia cho 0

        # === Buoc 3: Tru hip center va chia cho body size ===
        xy_normalized = (xy - hip_center) / body_size

        # === Buoc 4: Xu ly diem co confidence thap ===
        # Dat toa do ve 0 neu confidence qua thap (diem khong nhin thay)
        low_conf_mask = conf < DEFAULT_MIN_CONFIDENCE
        xy_normalized[low_conf_mask] = 0

        normalized[i, :, :2] = xy_normalized

    return normalized


def create_sliding_windows(data, label, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE):
    """
    Cat chuoi frame thanh cac window nho de train model

    Vi du: video co 100 frame, window_size=30, stride=10
    -> Window 1: frame 0-29
    -> Window 2: frame 10-39
    -> Window 3: frame 20-49
    -> ...

    Args:
        data: numpy array (num_frames, 17, 3) da chuan hoa
        label: 0 (normal) hoac 1 (fall)
        window_size: So frame moi window
        stride: Buoc nhay giua cac window

    Returns:
        list cac tuple (window_data, label)
        Moi window_data co shape (window_size, 17, 3)
    """
    windows = []
    num_frames = data.shape[0]

    # Video qua ngan thi bo qua
    if num_frames < DEFAULT_MIN_FRAMES:
        return windows

    # Neu video ngan hon window_size, pad bang cach lap frame cuoi
    if num_frames < window_size:
        # Pad bang cach lap lai frame cuoi cung
        pad_frames = window_size - num_frames
        last_frame = data[-1:]  # (1, 17, 3)
        padding = np.repeat(last_frame, pad_frames, axis=0)
        data = np.concatenate([data, padding], axis=0)
        windows.append((data, label))
        return windows

    # Tao sliding windows
    for start in range(0, num_frames - window_size + 1, stride):
        window = data[start:start + window_size]  # (window_size, 17, 3)
        windows.append((window, label))

    return windows


def flatten_window(window):
    """
    Lam phang window tu (window_size, 17, 3) thanh (window_size, 51)
    De lam input cho LSTM/TCN

    Args:
        window: numpy array (window_size, 17, 3)

    Returns:
        numpy array (window_size, 51)
    """
    return window.reshape(window.shape[0], -1)


def build_dataset(dataset_dir, output_dir, window_size=DEFAULT_WINDOW_SIZE,
                  stride=DEFAULT_STRIDE, limit=0):
    """
    Ham chinh: xay dung skeleton dataset tu CSV files

    Args:
        dataset_dir: Duong dan thu muc dataset (chua Fall/ va No_Fall/)
        output_dir: Duong dan thu muc luu .npy files
        window_size: So frame moi window
        stride: Buoc nhay giua cac window
        limit: Gioi han so video moi nhan (0 = tat ca)
    """
    fall_csv_dir = os.path.join(dataset_dir, 'Fall', 'Keypoints_CSV')
    nofall_csv_dir = os.path.join(dataset_dir, 'No_Fall', 'Keypoints_CSV')

    # Tim tat ca file CSV
    fall_csvs = sorted(glob.glob(os.path.join(fall_csv_dir, '*.csv')))
    nofall_csvs = sorted(glob.glob(os.path.join(nofall_csv_dir, '*.csv')))

    print("=" * 60)
    print("  XAY DUNG SKELETON DATASET")
    print("=" * 60)
    print(f"  Thu muc dataset : {dataset_dir}")
    print(f"  Thu muc output  : {output_dir}")
    print(f"  Window size     : {window_size} frames")
    print(f"  Stride          : {stride} frames")
    print(f"  So CSV Fall     : {len(fall_csvs)}")
    print(f"  So CSV No_Fall  : {len(nofall_csvs)}")
    print("=" * 60)
    print()

    # Gioi han so video neu can (de test nhanh)
    if limit > 0:
        fall_csvs = fall_csvs[:limit]
        nofall_csvs = nofall_csvs[:limit]
        print(f"[INFO] Gioi han: {limit} video moi nhan")
        print()

    # === BUOC 1: Doc va xu ly tung video ===
    all_windows = []    # List tat ca windows
    all_labels = []     # Label tuong ung
    stats = {
        'fall_videos_ok': 0,
        'fall_videos_fail': 0,
        'nofall_videos_ok': 0,
        'nofall_videos_fail': 0,
        'fall_windows': 0,
        'nofall_windows': 0,
        'skipped_short': 0
    }

    # --- Xu ly Fall videos (label = 1) ---
    print("[1/4] Dang xu ly Fall videos...")
    for idx, csv_path in enumerate(fall_csvs):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  Dang xu ly: {idx + 1}/{len(fall_csvs)}")

        # Doc CSV
        data = parse_csv(csv_path)
        if data is None or data.shape[0] < DEFAULT_MIN_FRAMES:
            if data is not None and data.shape[0] < DEFAULT_MIN_FRAMES:
                stats['skipped_short'] += 1
            stats['fall_videos_fail'] += 1
            continue

        # Chuan hoa skeleton
        data_norm = normalize_skeleton(data)

        # Tao sliding windows
        windows = create_sliding_windows(data_norm, label=1,
                                         window_size=window_size, stride=stride)
        if len(windows) == 0:
            stats['fall_videos_fail'] += 1
            continue

        # Flatten va them vao dataset
        for window, label in windows:
            all_windows.append(flatten_window(window))
            all_labels.append(label)

        stats['fall_videos_ok'] += 1
        stats['fall_windows'] += len(windows)

    print(f"  -> Xong! {stats['fall_videos_ok']} videos OK, "
          f"{stats['fall_videos_fail']} fail, "
          f"{stats['fall_windows']} windows")
    print()

    # --- Xu ly No_Fall videos (label = 0) ---
    print("[2/4] Dang xu ly No_Fall videos...")
    for idx, csv_path in enumerate(nofall_csvs):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  Dang xu ly: {idx + 1}/{len(nofall_csvs)}")

        data = parse_csv(csv_path)
        if data is None or data.shape[0] < DEFAULT_MIN_FRAMES:
            if data is not None and data.shape[0] < DEFAULT_MIN_FRAMES:
                stats['skipped_short'] += 1
            stats['nofall_videos_fail'] += 1
            continue

        data_norm = normalize_skeleton(data)

        windows = create_sliding_windows(data_norm, label=0,
                                         window_size=window_size, stride=stride)
        if len(windows) == 0:
            stats['nofall_videos_fail'] += 1
            continue

        for window, label in windows:
            all_windows.append(flatten_window(window))
            all_labels.append(label)

        stats['nofall_videos_ok'] += 1
        stats['nofall_windows'] += len(windows)

    print(f"  -> Xong! {stats['nofall_videos_ok']} videos OK, "
          f"{stats['nofall_videos_fail']} fail, "
          f"{stats['nofall_windows']} windows")
    print()

    if len(all_windows) == 0:
        print("[LOI] Khong co window nao duoc tao! Kiem tra lai dataset.")
        return

    # === BUOC 2: Chuyen thanh numpy array ===
    print("[3/4] Dang chuyen doi va chia train/val/test...")
    X = np.array(all_windows)  # (num_samples, window_size, 51)
    y = np.array(all_labels)   # (num_samples,)

    print(f"  Tong so windows: {len(X)}")
    print(f"  Shape moi window: {X[0].shape}")
    print(f"  Fall windows: {np.sum(y == 1)}")
    print(f"  Normal windows: {np.sum(y == 0)}")

    # === BUOC 3: Chia train/val/test ===
    # Shuffle truoc khi chia
    indices = np.arange(len(X))
    np.random.seed(42)  # Co dinh seed de ket qua on dinh
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Chia 70% train, 15% val, 15% test
    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    # === BUOC 4: Luu ra file .npy ===
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Dang luu file .npy vao {output_dir}...")

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    print()
    print("=" * 60)
    print("  KET QUA XAY DUNG DATASET")
    print("=" * 60)
    print(f"  Train: {len(X_train)} samples "
          f"(fall={np.sum(y_train == 1)}, normal={np.sum(y_train == 0)})")
    print(f"  Val  : {len(X_val)} samples "
          f"(fall={np.sum(y_val == 1)}, normal={np.sum(y_val == 0)})")
    print(f"  Test : {len(X_test)} samples "
          f"(fall={np.sum(y_test == 1)}, normal={np.sum(y_test == 0)})")
    print(f"  Shape: ({window_size}, {X_train.shape[2]})")
    print("-" * 60)
    print(f"  Videos Fall OK   : {stats['fall_videos_ok']}")
    print(f"  Videos NoFall OK : {stats['nofall_videos_ok']}")
    print(f"  Videos bo qua    : {stats['skipped_short']} (qua ngan)")
    print(f"  Videos loi       : {stats['fall_videos_fail'] + stats['nofall_videos_fail']}")
    print("=" * 60)
    print()
    print("Dataset da san sang de train LSTM/TCN!")
    print(f"Thu muc chua dataset: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Xay dung skeleton dataset tu CSV keypoints'
    )
    parser.add_argument(
        '--dataset', type=str,
        default=r'C:\Download\Do_an\dataset\video_fall',
        help='Duong dan thu muc dataset (chua Fall/ va No_Fall/)'
    )
    parser.add_argument(
        '--output', type=str,
        default='datasets/skeleton_windows',
        help='Thu muc luu file .npy (mac dinh: datasets/skeleton_windows)'
    )
    parser.add_argument(
        '--window-size', type=int, default=DEFAULT_WINDOW_SIZE,
        help=f'So frame moi window (mac dinh: {DEFAULT_WINDOW_SIZE})'
    )
    parser.add_argument(
        '--stride', type=int, default=DEFAULT_STRIDE,
        help=f'Buoc nhay giua cac window (mac dinh: {DEFAULT_STRIDE})'
    )
    parser.add_argument(
        '--limit', type=int, default=0,
        help='Gioi han so video moi nhan de test nhanh (0 = tat ca)'
    )

    args = parser.parse_args()

    build_dataset(
        dataset_dir=args.dataset,
        output_dir=args.output,
        window_size=args.window_size,
        stride=args.stride,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
