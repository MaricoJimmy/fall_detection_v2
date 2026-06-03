"""
Script cài đặt tự động
"""
import subprocess
import sys
import os

def install_requirements():
    """Cài đặt các thư viện cần thiết"""
    print("=" * 60)
    print("  CAI DAT THU VIEN CHO FALL DETECTION SYSTEM")
    print("=" * 60)
    print()

    requirements = [
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "opencv-contrib-python>=4.8.0",
        "ultralytics>=8.0.0",
        "mediapipe>=0.10.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.0"
    ]

    print("Dang cai dat cac thu vien sau:")
    for req in requirements:
        print(f"  - {req}")

    print()
    print("-" * 60)

    for req in requirements:
        print(f"\n[+] Dang cai dat {req}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", req
            ])
            print(f"    -> Thanh cong!")
        except subprocess.CalledProcessError as e:
            print(f"    -> Loi: {e}")
            print(f"    -> Vui long cai dat thu cong: pip install {req}")

    print()
    print("=" * 60)
    print("  CAI DAT HOAN TAT!")
    print("=" * 60)
    print()
    print("Ban co the chay demo bang:")
    print("  python demo_quick_start.py")
    print()

def check_camera():
    """Kiểm tra camera"""
    print("\nKiem tra camera...")
    import cv2

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Camera 0: OK")
        ret, frame = cap.read()
        if ret:
            print(f"Do phan giai: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
    else:
        print("Camera 0: KHONG TIM THAY")
        print("Vui long kiem tra ket noi camera!")

if __name__ == "__main__":
    install_requirements()
    check_camera()