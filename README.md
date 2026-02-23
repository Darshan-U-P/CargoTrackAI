# CargoTrackAI

CargoTrackAI is a computer vision project that demonstrates how deep
learning--based object detection can be integrated into a structured
object counting system using rule-based motion analysis.

This project covers the complete pipeline:

-   Extracting frames from video
-   Labeling images using annotation tools
-   Training a custom YOLO model via command line
-   Running inference on video using a Python script
-   Counting detected objects using motion logic

The implementation is fully done in Python.

------------------------------------------------------------------------

## 📁 Project Structure

    CargoTrackAI/
    │
    ├── dataset/            # Labeled dataset (images + labels)
    ├── original data/      # Raw source videos
    ├── train/              # Training configuration files
    ├── runs/detect/        # YOLO training outputs
    ├── objects/            # Extracted frames or processed images
    │
    ├── extract_frames.py   # Script to extract images from video
    ├── test_model.py       # Script to run inference + counting
    ├── req.txt             # Python dependencies
    ├── README.md
    ├── LICENSE
    └── .gitignore

------------------------------------------------------------------------

# 🚀 Workflow Overview

## 1️⃣ Step 1 -- Extract Frames from Video

Raw videos were placed inside:

    original data/

Frames were extracted using:

``` bash
python extract_frames.py
```

This script: - Reads video using OpenCV - Extracts frames at a fixed
interval - Saves images into the dataset folder

------------------------------------------------------------------------

## 2️⃣ Step 2 -- Image Labeling

After extracting frames, images were labeled using an annotation tool.

### Tools You Can Use

-   Roboflow (Web-based)
-   LabelImg (Desktop)
-   CVAT
-   Any YOLO-compatible labeling tool

### Label Format

Images were labeled in **YOLO format**:

    class_id x_center y_center width height

All values are normalized between 0 and 1.

The dataset folder follows this structure:

    dataset/
     ├── images/
     │    ├── train/
     │    └── val/
     └── labels/
          ├── train/
          └── val/

------------------------------------------------------------------------

## 3️⃣ Step 3 -- Model Training (Command Line)

Training was done using Ultralytics YOLO via command line.

Example training command:

``` bash
yolo detect train   model=yolov8n.pt   data=train/data.yaml   epochs=50   imgsz=640
```

After training completes, model weights are saved in:

    runs/detect/train/weights/best.pt

------------------------------------------------------------------------

## 4️⃣ Step 4 -- Testing the Model (Manual Video Input)

Inference is performed using:

``` bash
python test_model.py
```

This script:

-   Loads the trained model
-   Accepts manual video input
-   Runs object detection
-   Tracks object movement
-   Applies rule-based counting logic
-   Displays processed video with:
    -   Bounding boxes
    -   Counting lines
    -   Live object count

------------------------------------------------------------------------

# 🧠 Counting Logic

The system uses two horizontal reference lines:

-   Lower line → Entry zone
-   Upper line → Exit zone

Counting rule:

1.  Object crosses lower line → marked as inside
2.  Object crosses upper line → count incremented
3.  Each object is counted only once

This prevents duplicate counting.

------------------------------------------------------------------------

# 🛠 Installation

### 1. Clone Repository

``` bash
git clone https://github.com/Darshan-U-P/CargoTrackAI.git
cd CargoTrackAI
```

### 2. Install Dependencies

``` bash
pip install -r req.txt
```

------------------------------------------------------------------------

# ▶️ Run Inference

``` bash
python test_model.py
```

Make sure your trained model path inside `test_model.py` is correct.

------------------------------------------------------------------------

# 📦 Requirements

Major dependencies:

-   Python 3.9+
-   ultralytics
-   opencv-python
-   torch
-   numpy

See `req.txt` for full list.

------------------------------------------------------------------------

# 🎯 Purpose

CargoTrackAI demonstrates:

-   End-to-end custom object detection training
-   Practical dataset preparation
-   Real-time video inference
-   Rule-based object counting logic
-   Structured computer vision pipeline implementation

This project is designed for learning, experimentation, and
demonstration of applied AI in monitoring systems.

------------------------------------------------------------------------

# 📜 License

This project is licensed under the MIT License.
