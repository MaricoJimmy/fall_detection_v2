"""
Test he thong - Kiem tra cac module
"""
import sys
import os

# Test imports
print("=" * 60)
print("  KIEM TRA HE THONG FALL DETECTION")
print("=" * 60)
print()

# Test 1: Check Python version
print("[1] Python Version:")
print(f"    {sys.version}")
print()

# Test 2: Check required packages
print("[2] Checking required packages:")

required = {
    'numpy': 'NumPy',
    'cv2': 'OpenCV',
    'ultralytics': 'YOLOv8',
    'mediapipe': 'MediaPipe',
    'PIL': 'Pillow',
    'scipy': 'SciPy'
}

missing = []
for module, name in required.items():
    try:
        __import__(module)
        print(f"    [OK] {name}")
    except ImportError:
        print(f"    [MISSING] {name}")
        missing.append(name)

if missing:
    print()
    print("Thieu cac goi: " + ", ".join(missing))
    print("Chay: pip install -r requirements.txt")
else:
    print()
    print("Tat ca goi da duoc cai dat!")

# Test 3: Check camera
print()
print("[3] Checking camera:")
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("    [OK] Camera 0")
    cap.release()
else:
    print("    [WARNING] Camera 0 not found")

# Test 4: Check YOLO
print()
print("[4] Testing YOLOv8:")
try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print("    [OK] YOLOv8n loaded")
except Exception as e:
    print(f"    [ERROR] {e}")
    print("    YOLOv8 will auto-download when running demo")

# Test 5: Check MediaPipe
print()
print("[5] Testing MediaPipe Pose:")
try:
    import mediapipe as mp
    pose = mp.solutions.pose.Pose()
    print("    [OK] MediaPipe Pose initialized")
    pose.close()
except Exception as e:
    print(f"    [ERROR] {e}")

print()
print("=" * 60)
print("  KIEM TRA HOAN TAT!")
print("=" * 60)
print()

if len(missing) == 0:
    print("He thong san sang! Chay:")
    print("  python demo_quick_start.py")
else:
    print("Cai dat cac goi thieu truoc:")
    print("  pip install -r requirements.txt")