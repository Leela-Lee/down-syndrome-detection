# Deep Learning-Based Classification and Detection of Nuchal Translucency in Prenatal Ultrasound Images

> **A two-stage deep learning pipeline for automated Nuchal Translucency (NT) detection and Down Syndrome risk classification in first-trimester fetal ultrasound images.**

---

## Abstract

Down Syndrome (Trisomy 21) is one of the most common chromosomal abnormalities. Nuchal Translucency (NT) — a fluid-filled space at the back of a fetus's neck — is the standard first-trimester prenatal screening marker, measured via ultrasound between 11–14 weeks of pregnancy. Manual NT assessment is time-consuming, operator-dependent, and inaccessible in resource-limited settings.

This project presents a two-stage deep learning pipeline:

1. **Stage 1 — Classification:** A custom CNN classifies fetal ultrasound planes as *Standard* or *Non-Standard* — **89% accuracy**
2. **Stage 2 — NT Detection:** YOLOv8 object detection localises the NT bounding box — **mAP@0.5: 96.8%, Precision: 94.6%, Recall: 95.7%**

YOLOv8 outperforms YOLOv5 on all detection metrics, and the full pipeline assists clinicians in automated, reproducible prenatal NT assessment.

---

## Table of Contents

- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Demo / Inference](#demo--inference)
- [Requirements](#requirements)
- [References](#references)

---

## Dataset

- **Source:** 2D sagittal-view fetal ultrasound images — Shenzhen People's Hospital
- **Total images:** 1,528 (1,519 pregnant females)
- **Annotations:** `ObjectDetection.xlsx` — bounding box labels for 9 fetal structures including NT, Nasal Bone, Thalami, Midbrain, Palate, 4th Ventricle, Cisterna Magna, Nasal Tip, Nasal Skin
- **External test set:** 156 images — Longhua branch, Shenzhen People's Hospital

### Classification Split

| Class | Images |
|-------|--------|
| Standard | 812 |
| Non-Standard | Variable |
| Internal Test Set | 156 (72 Standard, 84 Non-Standard) |

### Detection Split (NT-annotated images)

| | Count |
|--|-------|
| Total NT-annotated | 1,110 |
| After deduplication | 1,074 |
| Train / Val / Test | 80 : 10 : 10 |

---

## Methodology

### Preprocessing (Both Stages)

| Step | Detail |
|------|--------|
| Grayscale conversion | `cv2.IMREAD_GRAYSCALE` |
| CLAHE enhancement | `clipLimit=2.0`, `tileGridSize=(8×8)` |
| Gaussian denoising | `kernel=(3×3)` |
| Resize | `225×225` px (classification) / `512×512` px (detection) |
| RGB conversion | Grayscale → 3-channel merge |

### Stage 1 — CNN Classification

- **Input:** Preprocessed ultrasound image `(512, 512, 3)`
- **Task:** Binary classification — Standard (1) vs Non-Standard (0)
- **Training:** 80/20 train-val split; final model trained for up to 150 epochs with early stopping

**Final model config:**
```
Data Augmentation: RandomFlip, RandomRotation(0.1), RandomZoom(0.1), RandomContrast(0.1)
Conv blocks: 4× [Conv2D → BatchNorm → ReLU → Dropout(0.1) → MaxPool]
Filters: 32 → 64 → 128 → 256
GlobalAveragePooling2D
Dense(128) → BatchNorm → ReLU → Dropout(0.4)
Dense(2, softmax)
Optimizer: Adam(lr=1e-3, clipvalue=1.0)
Loss: Categorical Crossentropy
L2 regularization: 1e-4
Callbacks: EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=5)
```

### Stage 2 — YOLOv8 NT Detection

**Label format:** YOLO normalised — `class_id x_center y_center width height`

**Data YAML:**
```yaml
path: /path/to/yolo_dataset
train: images/train
val: images/val
test: images/test
names:
  0: NT
```

**Training config:**
```
Model     : YOLOv8n (pretrained on COCO)
Epochs    : 50
imgsz     : 640
Batch     : 32
Confidence: 0.7
IoU       : 0.75
```

---

## Model Architecture

### CNN Classifier

```
Input (512, 512, 3)
├── Data Augmentation
├── Conv2D(32) → BN → ReLU → Dropout(0.1) → MaxPool
├── Conv2D(64) → BN → ReLU → Dropout(0.1) → MaxPool
├── Conv2D(128) → BN → ReLU → Dropout(0.1) → MaxPool
├── Conv2D(256) → BN → ReLU → Dropout(0.1) → MaxPool
├── GlobalAveragePooling2D
├── Dense(128) → BN → ReLU → Dropout(0.4)
└── Dense(2, softmax)
```

### YOLOv8n Detector

- **Backbone:** CSPDarknet (nano)
- **Neck:** PANet feature pyramid
- **Head:** Decoupled detection head
- **Class:** NT (single class)

---

## Results

### Stage 1 — Classification (CNN)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Non-Standard (0) | 0.92 | 0.87 | 0.90 | 84 |
| Standard (1) | 0.86 | 0.92 | 0.89 | 72 |
| **Overall Accuracy** | | | **0.89** | **156** |
| Macro Average | 0.89 | 0.89 | 0.89 | 156 |

### Stage 2 — Detection Comparison

| Metric | YOLOv5 | YOLOv8 |
|--------|--------|--------|
| Precision | 0.899 | **0.946** |
| Recall | 0.942 | **0.957** |
| mAP@0.5 | 0.962 | **0.968** |
| mAP@0.5:0.95 | 0.543 | **0.673** |
| Box Loss | 0.02414 | 1.09107 |

> YOLOv8 outperforms YOLOv5 across all primary detection metrics, particularly mAP@0.5:0.95 (+13%).

---

## Project Structure

```
down-syndrome-detection/
│
├── down-syndrome-yolo.ipynb       # Stage 2: NT detection (YOLOv8 + YOLOv5)
├── down-syndrome-class.ipynb      # Stage 1: Standard/Non-Standard classification (CNN)
├── data.yaml                      # YOLOv8 dataset config
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── NT-Detection/
│   └── weights/best.pt            # Best YOLOv8 weights
│
└── README.md
```

---

## Setup & Installation

```bash
git clone https://github.com/<your-username>/down-syndrome-detection.git
cd down-syndrome-detection

pip install ultralytics --upgrade
pip install tensorflow opencv-python matplotlib scikit-learn seaborn tqdm
```

---

## Usage

### Train CNN Classifier

Run all cells in `down-syndrome-class.ipynb`. Ensure dataset paths point to your `Standard` / `Non-Standard` image folders.

### Train YOLOv8 Detector

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    project="NT-Detection"
)
```

---

## Demo / Inference

### NT Detection

```python
from ultralytics import YOLO
import cv2

model = YOLO("NT-Detection/weights/best.pt")
results = model.predict(source="path/to/ultrasound.png", imgsz=640, conf=0.7, iou=0.75)

for r in results:
    img = r.plot()
    cv2.imshow("NT Detection", img)
    cv2.waitKey(0)
```

### Risk Classification

```python
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("cnn_classifier.h5")

img = cv2.imread("path/to/image.png", cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img = clahe.apply(img)
img = cv2.GaussianBlur(img, (3,3), 0)
img = cv2.resize(img, (512, 512))
img = cv2.merge([img, img, img])
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
label = "Standard" if np.argmax(pred) == 1 else "Non-Standard (DS Risk)"
print(f"Prediction: {label} | Confidence: {np.max(pred):.2f}")
```

---

## Requirements

```
ultralytics>=8.0
tensorflow>=2.10
opencv-python>=4.5
scikit-learn>=1.0
matplotlib>=3.5
seaborn
tqdm
numpy
```

---

## References

1. Kasera et al. "Deep-learning computer vision can identify increased nuchal translucency in the first trimester of pregnancy." *Prenatal Diagnosis*, 2024.
2. Reshi et al. "Deep Learning-Based Architecture for Down Syndrome Assessment During Early Pregnancy Using Fetal Ultrasound Images." *IJERR*, 2024.
3. Thomas & Resmi. "Computational Method of Predicting Down Syndrome on Foetus by Utilizing First Trimester Ultrasound Scan." *ICSCC*, 2023.
4. Saini. "Enhanced Detection of Down Syndrome Through CNNs: A Deep Learning Approach." *ICMCSI*, 2025.
5. Gokulakrishnan & Selvakumar. "An Efficient Approach for Detecting Down Syndrome Fetus Images Using Deep Learning Method." *IJCESE*, 2024.

---

## Author

**Leela Bhargavi N R**
MTech Data Science — Amrita Vishwa Vidyapeetham
[LinkedIn](https://www.linkedin.com/in/leela-bhargavi-n-r/) | leelab1113@gmail.com
