
# 🚦 AI Traffic Violation Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-red?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered system designed to automatically detect and log common **traffic violations** from recorded videos. 
---

## ✨ Our Solution

Traffic monitoring is mostly manual, error-prone, and inefficient.  
This system solves this by using **deep learning models** to detect vehicles and violations in real time and generate **evidence snapshots** with timestamps.

**Key features implemented:**
- 🚗 **Vehicle Detection** (Cars, Buses, Trucks, Motorcycles) using YOLOv8
- 🚦 **Stop-Line Crossing Detection** (Red Light Jumping logic)
- 📸 **Evidence Generation** (Snapshots + CSV Logs)
- 🔢 **License Plate Recognition** (In-progress using EasyOCR)

---

## 🎥 Prototype Showcase

I implemented a working prototype to demonstrate real-world use cases:
- Playback analysis on **video files**
- Automatic snapshot storage for **violation reports**
- Output video generation with bounding boxes and stop-line overlay

---

## ✅ Core Features Implemented

- **YOLOv8 Object Detection** → Detects cars, buses, trucks, and motorcycles
- **Centroid Tracking** → Tracks objects across frames for violation logic
- **Violation Logic Module** → Applies custom rules (stop-line crossing)
- **Evidence Generation** → Saves violation images with timestamp and label
- **(Optional) LPR Module** → License Plate Recognition using EasyOCR

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** OpenCV, NumPy, Pandas, EasyOCR, Ultralytics
- **Detection Model:** YOLOv8n (pre-trained on COCO dataset)
- **Tracking:** Custom Centroid Tracker
- **Environment:** Google Colab

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/mukarma1/TrafficHawk-AI-Detection.git

# Move into the project folder
cd TrafficHawk-AI-Detection

# Install dependencies
pip install -r requirements.txt
Download YOLOv8 Weights
bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
Run Detection
bash
python colab_detect.py
Note: This project was executed in Google Colab. The script colab_detect.py is optimized for Colab (no GUI/webcam support). For local execution, minor modifications may be required.

📊 Results
Processed 6401 frames from a test video

Successfully detected 30+ violations (cars, buses, trucks)

Generated evidence images with timestamps

Output video with bounding boxes and stop-line overlay (output_small.mp4)

Logged all violations in violations_log.csv

👤 Author
[Mukarma Rehman]

GitHub: https://github.com/mukarma1

LinkedIn: https://www.linkedin.com/in/mukarma-rehman-7b4571419/

📜 License
This project is licensed under the MIT License — you are free to use, modify, and distribute this project with attribution.

