# 🚦 AI Traffic Violation Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-red?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered system that detects vehicles in recorded traffic videos, flags **stop-line crossings**, and logs each one with timestamped evidence.

> **Scope note:** The system currently detects stop-line crossings only (logged as `line_crossing`). It does **not** read traffic signal color yet, so it cannot tell a red-light violation from a normal crossing. Signal-state detection is on the roadmap below.

---

## ✨ Our Solution

Traffic monitoring is mostly manual, error-prone, and inefficient.
This project uses a deep learning detector (YOLOv8) to find vehicles in recorded video, tracks them across frames, and generates **evidence snapshots with timestamps** whenever a vehicle crosses a defined stop line.

**Key features implemented:**

- 🚗 **Vehicle Detection** (Cars, Buses, Trucks, Motorcycles) using YOLOv8
- 🚦 **Stop-Line Crossing Detection** (`line_crossing` events)
- 📸 **Evidence Generation** (Snapshots + CSV Logs)
- 🔢 **License Plate Recognition** (In progress using EasyOCR)

---

## 🎥 Prototype Showcase

A working prototype that demonstrates the pipeline end to end:

- Playback analysis on **video files**
- Automatic snapshot storage for **violation reports**
- Output video with bounding boxes and stop-line overlay

---

## ✅ Core Features Implemented

- **YOLOv8 Object Detection** → Detects cars, buses, trucks, and motorcycles
- **Centroid Tracking** → Tracks objects across frames for the crossing logic
- **Violation Logic Module** → Applies a custom rule (stop-line crossing)
- **Evidence Generation** → Saves violation images with timestamp and label
- **(In progress) LPR Module** → License plate recognition using EasyOCR

---

## ⚠️ Known Limitations

- **No signal color detection:** the model never looks at the traffic light, so every stop-line crossing is logged, whether the light is red or not.
- **Tracking under occlusion:** the custom centroid tracker matches objects by the distance between box centers, so IDs can swap or get lost when vehicles overlap. Since the stop-line check depends on tracked IDs, this can cause missed or duplicate events.
- **Recorded video only:** tested on pre-recorded video in Google Colab, not on a live camera stream.

## 🗺️ Roadmap

- [ ] Traffic light state check (HSV color filtering on the signal region, or a small classifier) so violations are flagged only when the light is red
- [ ] Replace the centroid tracker with ByteTrack or DeepSORT for more stable IDs during occlusion
- [ ] Finish the license plate recognition module
- [ ] Human-in-the-loop review step for flagged events (exceptions like emergency vehicles or officer-directed traffic)

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
```

### Download YOLOv8 Weights

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Run Detection

```bash
python colab_detect.py
```

> **Note:** This project was executed in Google Colab. The script `colab_detect.py` is optimized for Colab (no GUI/webcam support). For local execution, minor modifications may be required.

---

## 📊 Results

- Processed **6401 frames** from a test video
- Logged **30+ stop-line crossings** (cars, buses, trucks)
- Generated evidence images with timestamps
- Output video with bounding boxes and stop-line overlay (`output_small.mp4`)
- All events saved in `violations_log.csv`

---

## 👤 Author

**Mukarma Rehman**

- GitHub: [mukarma1](https://github.com/mukarma1)
- LinkedIn: [Mukarma Rehman](https://www.linkedin.com/in/mukarma-rehman-7b4571419/)

---

## 📜 License

This project is licensed under the MIT License — you are free to use, modify, and distribute this project with attribution.
