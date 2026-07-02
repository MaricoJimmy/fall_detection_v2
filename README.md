# Real-Time Fall Detection System 🚀

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8--Pose-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

A robust, real-time Vision-based Fall Detection System designed for healthcare and elderly monitoring. This project leverages **YOLOv8-Pose** for spatial skeletal extraction and a custom **Temporal Convolutional Network (FallTCN)** engineered with **Kinematic Velocity** features to accurately classify temporal actions and drastically minimize false alarms.

## ✨ Key Features
- **Single-Stage Pose Extraction:** Utilizes YOLOv8-Pose for highly robust 17-keypoint extraction, overcoming occlusion issues commonly found in legacy tools like MediaPipe.
- **Velocity Feature Engineering:** Computes explicit temporal velocity derivatives from skeletal coordinates to mathematically distinguish between a dangerous fall and a safe horizontal sleeping posture.
- **TCN Sequence Modeling:** Replaces traditional LSTMs with a Temporal Convolutional Network. Dilated causal convolutions process 30-frame sequence windows in parallel, preventing memory loss (vanishing gradient) and ensuring ultra-fast inference.
- **Real-Time Performance:** Achieves ~30 FPS on standard GPU hardware, satisfying the low-latency requirements for medical early warning systems.

## 📊 Performance Benchmark
The system was iteratively optimized. The final `FallTCN (Velocity)` model successfully solved the False Alarm bottleneck that plagued baseline models.

| Architecture Stack | Accuracy | Recall (Sensitivity) | F1-Score | False Alarm Rate |
| :--- | :---: | :---: | :---: | :---: |
| MediaPipe + Rule-based | 56.67% | 16.13% | 27.78% | 0.00% |
| MediaPipe + LSTM | 38.33% | 48.39% | 44.78% | > 70.0% |
| YOLOv8-Pose + Standard TCN | 72.97% | **100.0%** | 77.27% | 50.00% |
| **YOLOv8-Pose + FallTCN (Velocity)** | **84.51%** | 79.92% | **79.48%** | **12.72%** |

## 📁 Project Structure
```text
fall_detection_Project/
├── datasets/                 # Training datasets (UR Fall, etc.)
├── models/                   # Pre-trained weights (YOLO & TCN)
├── build_dataset_veloci.py   # Extracts keypoints & velocity into sliding windows
├── train_tcn_veloci.py       # Defines and trains the FallTCN model
├── test_video_veloci.py      # Real-time inference script with OpenCV UI
├── plot_metrics.py           # Evaluation visualization tools
└── requirements.txt          # Python dependencies
```

## ⚙️ Installation & Usage

### 1. Environment Setup
It is highly recommended to use a virtual environment with a CUDA-enabled PyTorch installation.
```bash
git clone https://github.com/MaricoJimmy/fall_detection_Project.git
cd fall_detection_Project
pip install -r requirements.txt
```

### 2. Model Preparation
Ensure you have the following weights inside the `models/` directory:
- `yolov8m-pose.pt` (Ultralytics Pre-trained)
- `TCN_Veloci_best.pth` (The custom trained FallTCN model)

### 3. Running Inference
To run the real-time detection on a pre-recorded video:
```bash
python test_video_veloci.py --input path/to/your/video.mp4
```
To run the inference using a live webcam feed:
```bash
python test_video_veloci.py --input 0
```

## 🧠 System Pipeline Overview
1. **Input:** Video stream (30 FPS).
2. **Spatial Extractor:** YOLOv8-Pose extracts the bounding box and 17 2D keypoints.
3. **Feature Engineering:** Calculates the $x, y$ velocity of each keypoint between frames.
4. **Windowing:** Aggregates coordinates and velocities into a `[30, 102]` spatial-temporal matrix.
5. **Temporal Modeler:** FallTCN processes the window using 1D Dilated Causal Convolutions.
6. **Output:** Triggers a red "FALL DETECTED" alert if the probability crosses the threshold.

## 📜 Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the state-of-the-art pose estimation framework.
- The authors of the [UR Fall Detection Dataset](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html) used during the fine-tuning phases.
