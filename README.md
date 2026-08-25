# Apple Object Detection & ByteTrack Video Counter

A complete Computer Vision pipeline for **Apple Detection, Multi-Object ByteTracking, & Real-time Apple Counting** using YOLO models (Ultralytics framework).

This repository provides scripts to convert COCO datasets, train/fine-tune models, run batch image inference with apple counts, track video streams with **ByteTrack** (to count unique apples without double-counting), stream real-time video/webcam detection, and evaluate model accuracy.

---

## 📁 Repository Structure

```
yolo26-apple/
├── weights/
│   └── best.pt               # Place your trained .pt model weights here
├── track_video.py            # ByteTrack Video Tracker & Cumulative Unique Apple Counter
├── convert_coco_to_yolo.py   # Converts COCO dataset to YOLO format & generates apple_dataset.yaml
├── train.py                  # Trains / fine-tunes YOLO model on Apple dataset
├── detect.py                 # Runs image inference & generates annotated outputs with count banner
├── webcam_detect.py          # Real-time webcam / video stream detection with live FPS & counter
├── evaluate.py               # Evaluates model performance (mAP50, mAP50-95, Precision, Recall)
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

## ⚡ Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Copy Your Custom Model Weights
Copy your trained model (`.pt`) file into the `weights/` directory:
```bash
cp /path/to/your/model.pt weights/best.pt
```

---

## 🎯 ByteTrack Video Tracking & Unique Apple Counter

To track apples in a video file and count the **Total Unique Apples** (avoiding double-counting the same apple across multiple frames):

```bash
# Fast 10-second test with custom confidence (e.g. conf 0.15)
python3 track_video.py --source apple_video.mp4 --model "weights/best.pt" --max-seconds 10 --conf 0.15 --save-video

# Full ByteTrack video tracking
python3 track_video.py --source apple_video.mp4 --model "weights/best.pt" --save-video

# ByteTrack with ROI Line Crossing Counter
python3 track_video.py --source apple_video.mp4 --model "weights/best.pt" --enable-line --line-position 0.5 --save-video
```

---

## 🛠️ Additional Tools & Usage

### Step 1: Convert Dataset (COCO to YOLO)
Convert your Roboflow COCO dataset to YOLO format:
```bash
python3 convert_coco_to_yolo.py --coco-dir "/Volumes/MacHDD/Shinjuku/apple-Forked on 8-25-2026.coco" --output-dir "./apple_dataset"
```

### Step 2: Train Model
Fine-tune a YOLO model on the apple dataset (automatically uses Apple Silicon `mps`, `cuda`, or `cpu`):
```bash
python3 train.py --data "./apple_dataset/apple_dataset.yaml" --model "yolo11n.pt" --epochs 50 --batch 16
```

### Step 3: Run Object Detection on Images
Detect apples on single images or a directory of test images:
```bash
python3 detect.py --source "./apple_dataset/images/test" --model "weights/best.pt" --conf 0.25
```

### Step 4: Real-time Camera Stream Detection
Stream live camera feed with optional ByteTrack tracking:
```bash
python3 webcam_detect.py --source 0 --model "weights/best.pt" --tracker "bytetrack.yaml"
```

### Step 5: Evaluate Model Performance
Evaluate mAP50, mAP50-95, Precision, and Recall on validation/test sets:
```bash
python3 evaluate.py --data "./apple_dataset/apple_dataset.yaml" --model "weights/best.pt" --split val
```

---

## ⚙️ Hardware Acceleration Support
The code automatically detects hardware acceleration:
- **Apple Silicon (M1/M2/M3/M4)**: Uses `--device mps` via PyTorch Metal Performance Shaders.
- **Nvidia GPU**: Uses `--device cuda`.
- **CPU**: Fallback to `--device cpu`.


python3 track_video.py --source "https://www.youtube.com/watch?v=CyS0bDBi7Dw" --model "weights/best.pt" --save-video

yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" "https://www.youtube.com/watch?v=CyS0bDBi7Dw" -o apple_video.mp4

python3 track_video.py --source apple_video.mp4 --model "weights/best.pt" --max-seconds 10 --conf 0.15 --save-video

python3 track_video.py --source apple_video.mp4 --model "weights/best.pt" --max-seconds 30 --conf 0.01 --save-video

python3 track_video.py --source apple_video.mp4 --model "weights/best.pt" --imgsz 1280 --max-seconds 10 --conf 0.01 --save-video

python3 detect.py --source "./apple_dataset/images/new_test/22741850_m.jpg" --model "weights/best.pt" --imgsz 1280 --conf 0.01


