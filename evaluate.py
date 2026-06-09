import os
import glob
import cv2
import argparse
import time
from ultralytics import YOLO

import config
from main import FallDetectionSystem

# Tắt cảnh báo âm thanh và lưu ảnh khi chạy evaluate
config.ALERT_CONFIG['sound_enabled'] = False
config.ALERT_CONFIG['save_fall_images'] = False

def evaluate_video(system, video_path):
    """
    Chạy nhận diện trên một video để xem có phát hiện ngã hay không.
    Trả về True nếu CÓ phát hiện ngã, False nếu KHÔNG.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return False
        
    fall_detected = False
    
    # Reset fall detector để dọn dẹp trạng thái từ video trước
    system.fall_detector.reset()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break # Hết video
        # Lấy timestamp nội bộ của video
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Không cần dùng processed_frame vì ta chạy headless
        _, fall_info = system.process_frame(frame, timestamp=timestamp)
        # Nếu mô hình chẩn đoán là ngã
        if fall_info['status'] == 'FALL':
            fall_detected = True
            break # Dừng sớm video này để tiết kiệm thời gian vì đã phát hiện ngã
            
    cap.release()
    return fall_detected

def main():
    parser = argparse.ArgumentParser(description='Đánh giá mô hình Fall Detection')
    parser.add_argument('--dataset', type=str, required=True, help='Đường dẫn thư mục dataset (VD: C:/Download/Do an/archive)')
    parser.add_argument('--limit', type=int, default=0, help='Giới hạn số lượng video mỗi nhãn để test nhanh (0 = test toàn bộ)')
    
    args = parser.parse_args()
    
    dataset_dir = args.dataset
    fall_dir = os.path.join(dataset_dir, 'Fall', 'Raw_Video')
    no_fall_dir = os.path.join(dataset_dir, 'No_Fall', 'Raw_Video')
    
    if not os.path.exists(fall_dir) or not os.path.exists(no_fall_dir):
        print(f"Lỗi: Thư mục không hợp lệ! Vui lòng đảm bảo {fall_dir} và {no_fall_dir} tồn tại.")
        return
        
    fall_videos = glob.glob(os.path.join(fall_dir, '*.mp4')) + glob.glob(os.path.join(fall_dir, '*.avi'))
    no_fall_videos = glob.glob(os.path.join(no_fall_dir, '*.mp4')) + glob.glob(os.path.join(no_fall_dir, '*.avi'))
    
    if args.limit > 0:
        import random
        random.seed(42) # Giữ cố định random seed để kết quả ổn định giữa các lần test
        # Chọn ngẫu nhiên giới hạn số lượng video
        fall_videos = random.sample(fall_videos, min(args.limit, len(fall_videos)))
        no_fall_videos = random.sample(no_fall_videos, min(args.limit, len(no_fall_videos)))
        
    print("="*60)
    print("  BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH (HEADLESS MODE)")
    print(f"  Tập Fall (Có ngã): {len(fall_videos)} video")
    print(f"  Tập No_Fall (Bình thường): {len(no_fall_videos)} video")
    print("="*60)
    
    # Khởi tạo mô hình một lần duy nhất
    print("\nĐang khởi tạo các mô hình AI...")
    start_time = time.time()
    system = FallDetectionSystem()
    print("Khởi tạo xong! Bắt đầu quét các video...\n")
    
    # Thống kê
    TP = 0 # True Positive: Ngã -> Nhận diện ngã
    FN = 0 # False Negative: Ngã -> Không nhận diện
    TN = 0 # True Negative: Bình thường -> Không nhận diện ngã
    FP = 0 # False Positive: Bình thường -> Nhận diện nhầm là ngã
    
    failed_logs = []
    
    # 1. Đánh giá trên tập CÓ ngã (Mong đợi: True)
    try:
        from tqdm import tqdm
        iterable_fall = tqdm(fall_videos, desc="Đang quét tập CÓ ngã")
    except ImportError:
        iterable_fall = fall_videos
        
    for idx, video_path in enumerate(iterable_fall):
        if 'tqdm' not in sys.modules:
            print(f"Đang xử lý tập CÓ ngã [{idx+1}/{len(fall_videos)}]: {os.path.basename(video_path)}")
            
        detected = evaluate_video(system, video_path)
        if detected:
            TP += 1
        else:
            FN += 1
            failed_logs.append(f"MISS (FN): Bỏ sót ngã ở video {os.path.basename(video_path)}")
            
    # 2. Đánh giá trên tập KHÔNG ngã (Mong đợi: False)
    try:
        from tqdm import tqdm
        iterable_nofall = tqdm(no_fall_videos, desc="Đang quét tập KHÔNG ngã")
    except ImportError:
        iterable_nofall = no_fall_videos
        
    for idx, video_path in enumerate(iterable_nofall):
        if 'tqdm' not in sys.modules:
            print(f"Đang xử lý tập KHÔNG ngã [{idx+1}/{len(no_fall_videos)}]: {os.path.basename(video_path)}")
            
        detected = evaluate_video(system, video_path)
        if detected:
            FP += 1
            failed_logs.append(f"FALSE ALARM (FP): Nhận diện nhầm ở video {os.path.basename(video_path)}")
        else:
            TN += 1
            
    # Tính toán Metrics
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    end_time = time.time()
    
    # In báo cáo
    print("\n" + "="*60)
    print("  KẾT QUẢ ĐÁNH GIÁ (EVALUATION REPORT)")
    print("="*60)
    print(f"Tổng số video đã test: {total}")
    print(f"Thời gian chạy: {end_time - start_time:.1f} giây")
    print("-" * 60)
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(f"  - True Positive  (Bắt đúng ngã)       : {TP}")
    print(f"  - False Positive (Báo động giả)       : {FP}")
    print(f"  - True Negative  (Bình thường chuẩn)  : {TN}")
    print(f"  - False Negative (Bỏ sót ngã)         : {FN}")
    print("-" * 60)
    print(f"Accuracy  (Độ chính xác) : {accuracy*100:.2f}%")
    print(f"Precision (Độ chuẩn xác) : {precision*100:.2f}%")
    print(f"Recall    (Độ nhạy)      : {recall*100:.2f}%")
    print(f"F1-Score                 : {f1_score*100:.2f}%")
    print("=" * 60)
    
    # Ghi log lỗi
    log_file = "evaluation_errors.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("DANH SÁCH CÁC VIDEO NHẬN DIỆN SAI:\n")
        f.write("===================================\n")
        for log in failed_logs:
            f.write(log + "\n")
            
    print(f"\n[Info] Danh sách các video bị nhận diện sai đã được lưu vào file: {log_file}")
    print("[Info] Bạn có thể mở file này ra xem video nào bị lỗi để tinh chỉnh config.py")

if __name__ == '__main__':
    import sys
    main()
