import os
import cv2
import time
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------- CONFIG ----------------
YOLO_MODEL = "yolov8n.pt"
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

EVIDENCE_DIR = Path("evidence")
EVIDENCE_DIR.mkdir(exist_ok=True)
LOG_CSV = Path("violations_log.csv")
OUTPUT_VIDEO = Path("output.mp4")

MIN_VEHICLE_AREA = 500
MAX_TRACK_DISTANCE = 60
OCR_GPU = False

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("easyocr not available, skipping OCR")

# ---------------- Centroid Tracker ----------------
class CentroidTracker:
    def __init__(self, max_distance=MAX_TRACK_DISTANCE):
        self.next_id = 1
        self.objects = {}
        self.last_seen = {}
        self.tracks = {}
        self.max_distance = max_distance

    def update(self, detections):
        assigned = {}
        now = time.time()
        if not self.objects:
            for c in detections:
                oid = self.next_id; self.next_id += 1
                self.objects[oid] = c
                self.last_seen[oid] = now
                self.tracks[oid] = [c]
                assigned[oid] = c
            return assigned

        used_ids = set()
        for c in detections:
            best_id, best_dist = None, None
            for oid, cent in self.objects.items():
                if oid in used_ids: continue
                dist = math.hypot(c[0]-cent[0], c[1]-cent[1])
                if best_dist is None or dist < best_dist:
                    best_dist, best_id = dist, oid
            if best_dist is not None and best_dist <= self.max_distance:
                self.objects[best_id] = c
                self.last_seen[best_id] = now
                self.tracks[best_id].append(c)
                assigned[best_id] = c
                used_ids.add(best_id)
            else:
                oid = self.next_id; self.next_id += 1
                self.objects[oid] = c
                self.last_seen[oid] = now
                self.tracks[oid] = [c]
                assigned[oid] = c
                used_ids.add(oid)

        stale = [oid for oid, t in self.last_seen.items() if now - t > 5.0]
        for oid in stale:
            self.objects.pop(oid, None)
            self.last_seen.pop(oid, None)
            self.tracks.pop(oid, None)
        return assigned

# ---------------- Helpers ----------------
def draw_line(img, line, color=(0,0,255), thickness=2):
    (x1,y1,x2,y2) = map(int, line)
    cv2.line(img, (x1,y1), (x2,y2), color, thickness)

def point_side_of_line(pt, line):
    (x1,y1,x2,y2) = line
    x,y = pt
    return (x2-x1)*(y-y1) - (y2-y1)*(x-x1)

def centroid_from_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return (int((x1+x2)/2), int((y1+y2)/2))

def save_evidence(img, prefix="violation"):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    path = EVIDENCE_DIR / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(path), img)
    return str(path)

def append_log(row):
    df = pd.DataFrame([row])
    if LOG_CSV.exists():
        df.to_csv(LOG_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_CSV, mode='w', header=True, index=False)

# ---------------- Main ----------------
def run_detection(source):
    print(f"Opening source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = max(1, int(cap.get(cv2.CAP_PROP_FPS) or 20))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps}fps, {total} frames")

    stop_line = (0, int(height*0.6), width, int(height*0.6))

    model = YOLO(YOLO_MODEL)
    ocr = easyocr.Reader(['en'], gpu=OCR_GPU) if OCR_AVAILABLE else None
    tracker = CentroidTracker()
    violated_ids = set()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.resize(frame, (width, height))
        img = frame.copy()

        results = model.predict(img, imgsz=640, conf=0.35, verbose=False)
        r = results[0]
        boxes, class_ids = [], []
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0].item())
            if cls in VEHICLE_CLASSES:
                if (x2-x1)*(y2-y1) < MIN_VEHICLE_AREA:
                    continue
                boxes.append((x1,y1,x2,y2))
                class_ids.append(cls)

        centroids = [centroid_from_bbox(b) for b in boxes]
        assigned = tracker.update(centroids)
        cent_to_info = {centroid_from_bbox(b): (b,c) for b,c in zip(boxes, class_ids)}

        for oid, cent in assigned.items():
            info = cent_to_info.get(cent)
            if info is None:
                min_d, found = None, None
                for b, c in zip(boxes, class_ids):
                    cc = centroid_from_bbox(b)
                    d = math.hypot(cc[0]-cent[0], cc[1]-cent[1])
                    if min_d is None or d < min_d:
                        min_d, found = d, (b,c)
                if not found: continue
                bbox, cls = found
            else:
                bbox, cls = info

            x1,y1,x2,y2 = bbox
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"{VEHICLE_CLASSES.get(cls)} ID:{oid}",
                        (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            track = tracker.tracks.get(oid, [])
            if len(track) >= 2:
                prev, curr = track[-2], track[-1]
                sp = point_side_of_line(prev, stop_line)
                sc = point_side_of_line(curr, stop_line)
                if sp != 0 and sc != 0 and (sp*sc) < 0:
                    if oid not in violated_ids:
                        pad = 20
                        xx1=max(0,x1-pad); yy1=max(0,y1-pad)
                        xx2=min(width-1,x2+pad); yy2=min(height-1,y2+pad)
                        crop = frame[yy1:yy2, xx1:xx2]
                        plate_text = None
                        if ocr is not None and crop.size > 0:
                            try:
                                res = ocr.readtext(crop[crop.shape[0]//2:, :])
                                texts = [t[1] for t in res if len(t[1]) >= 4]
                                if texts:
                                    plate_text = sorted(texts, key=len, reverse=True)[0]
                            except Exception:
                                pass

                        img_path = save_evidence(crop, prefix=f"cross_{VEHICLE_CLASSES.get(cls)}")
                        row = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "object_id": oid,
                            "class": VEHICLE_CLASSES.get(cls),
                            "plate": plate_text or "",
                            "image": img_path,
                            "reason": "line_crossing"
                        }
                        append_log(row)
                        print("Logged:", row)
                        violated_ids.add(oid)

        draw_line(img, stop_line)
        writer.write(img)

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Frame {frame_count}/{total} | {frame_count/elapsed:.1f} fps")

    cap.release()
    writer.release()
    print(f"\n✅ Done! Processed {frame_count} frames.")
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"Evidence images: {EVIDENCE_DIR}/")
    print(f"Log CSV: {LOG_CSV}")

if __name__ == "__main__":
    run_detection("sample.mp4")
