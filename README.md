# Prenatal Down Syndrome Detection via Nuchal Translucency Analysis in Fetal Ultrasound

> **A two-stage deep learning pipeline for automated Nuchal Translucency (NT) region detection and Down Syndrome risk classification in first-trimester fetal ultrasound images.**

---

## Abstract

Down Syndrome (Trisomy 21) is one of the most common chromosomal abnormalities. Nuchal Translucency (NT) measurement during the first trimester is the standard prenatal screening marker, but manual assessment is time-consuming, operator-dependent, and inaccessible in low-resource settings.

This project presents a two-stage deep learning pipeline:

1. **Stage 1 — NT Detection:** YOLOv8n object detection localises the NT region in raw fetal ultrasound images — **96% bounding-box accuracy**
2. **Stage 2 — Risk Classification:** A custom CNN classifies the NT plane as *Standard* (normal) or *Non-Standard* (abnormal/DS risk) — **90% classification accuracy**

The system assists clinicians by providing fast, automated, and reproducible NT assessment — reducing operator dependency and enabling broader prenatal screening access.

---

## Table of Contents

- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Demo / Inference](#demo--inference)
- [Requirements](#requirements)
- [Citation](#citation)

---

## Dataset

### Stage 1 — Detection
| Split | Source |
|-------|--------|
| Train | Kaggle — `object-detection/Train/ProcessedData` |
| Test  | Kaggle — `object-detction-test/Test/Images` |
| Format | PNG images + `.txt` bounding box annotations |
| Class | `NT` (Nuchal Translucency region) |
| Train/Val split | 80% / 20% (random shuffle) |

### Stage 2 — Classification
| Split | Source |
|-------|--------|
| Train | Kaggle — `classification/New folder/Train - Copy/` |
| Test  | Kaggle — `classification/New folder/Internal Test Set/` |
| Classes | `Standard` (label 1) / `Non-Standard` (label 0) |

> **Note:** Datasets contain private/clinical fetal ultrasound scans. Access is subject to original Kaggle dataset licenses.

---

## Methodology

### Shared Preprocessing (Both Stages)

Applied identically to all splits before training:

| Step | Detail |
|------|--------|
| Grayscale conversion | `cv2.IMREAD_GRAYSCALE` |
| CLAHE enhancement | `clipLimit=2.0`, `tileGridSize=(8,8)` — improves NT visibility in low-contrast scans |
| Gaussian denoising | Kernel `(3,3)` — reduces ultrasound speckle noise |
| Resize | `512×512` px |
| 3-channel conversion | Grayscale → RGB (`cv2.merge`) for model compatibility |

---

### Stage 1 — YOLOv8 NT Detection

#### Label Conversion
Original annotations in `[class, y_min, x_min, y_max, x_max]` (pixel coordinates) are converted to YOLO normalised format:

```
class_id  x_center  y_center  width  height
```

All values normalised by `IMAGE_WIDTH = IMAGE_HEIGHT = 512`.

**YAML config (`data.yaml`):**
```yaml
path: /path/to/yolo_dataset
train: images/train
val: images/val
test: images/test
names:
  0: NT
```

#### Two-Phase Training Strategy
Small anatomical structures like NT benefit from progressive resolution training:

| | Phase 1 — Coarse | Phase 2 — Fine-Grained |
|--|------------------|------------------------|
| **Weights** | `yolov8n.pt` (COCO pretrained) | Phase 1 `best.pt` |
| **Epochs** | 50 | 100 |
| **imgsz** | 640 | 1280 |
| **Batch** | 16 | 8 |
| **Optimizer** | SGD, `lr=0.001` | SGD, `lr=0.001` |
| **Early stopping** | `patience=30` | `patience=30` |

---

### Stage 2 — CNN Risk Classification

#### Data Augmentation (Final Model)
```python
RandomFlip("horizontal")
RandomRotation(0.1)
RandomZoom(0.1)
RandomContrast(0.1)
```

#### Hyperparameter Tuning Progression

Three iterations were explored before arriving at the final model:

| Version | Key Changes | Epochs |
|---------|------------|--------|
| **v1** — Baseline | 4× Conv2D (32→256) + GAP + Dense(2). Adam `lr=1e-4`, no regularisation | 50 |
| **v2** — BatchNorm + Dropout | Added BatchNormalization after each Conv. Dropout(0.5) before Dense. | 75 |
| **v3** — Full Regularisation | L2 (`1e-4`) on all layers, `dropout_conv=0.25`, `dropout_dense=0.6`, EarlyStopping + ReduceLROnPlateau | 150 |
| **Final** ✅ | BatchNorm + BN-after-Dense, `dropout_conv=0.1`, `dropout_dense=0.4`, `lr=1e-3`, `ReduceLROnPlateau(factor=0.5, patience=5)` | 150 |

---

## Model Architecture

### Stage 1 — YOLOv8n (Detection)

- **Backbone:** CSPDarknet (nano)
- **Neck:** PANet feature pyramid
- **Head:** Decoupled detection head
- **Input:** `640px` (Phase 1) → `1280px` (Phase 2)
- **Pretrained on:** COCO, fine-tuned on NT ultrasound data

### Stage 2 — CNN Classifier (Final Model)

```
Input (512, 512, 3)
│
├── Data Augmentation (RandomFlip, RandomRotation 0.1, RandomZoom 0.1, RandomContrast 0.1)
│
├── Conv2D(32, 3×3, same) → BatchNorm → ReLU → Dropout(0.1) → MaxPool(2×2)
├── Conv2D(64, 3×3, same) → BatchNorm → ReLU → Dropout(0.1) → MaxPool(2×2)
├── Conv2D(128, 3×3, same) → BatchNorm → ReLU → Dropout(0.1) → MaxPool(2×2)
├── Conv2D(256, 3×3, same) → BatchNorm → ReLU → Dropout(0.1) → MaxPool(2×2)
│
├── GlobalAveragePooling2D
├── Dense(128) → BatchNorm → ReLU → Dropout(0.4)
└── Dense(2, softmax)

Optimizer : Adam(lr=1e-3, clipvalue=1.0)
Loss      : Categorical Crossentropy
Callbacks : EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)
L2 Reg    : 1e-4 on all Conv + Dense layers
```

---

## Results

| Stage | Model | Metric | Score |
|-------|-------|--------|-------|
| NT Detection | YOLOv8n (Phase 2) | Bounding Box Accuracy | **96%** |
| Risk Classification | CNN (Final) | Accuracy | **90%** |
| Risk Classification | CNN (Final) | Precision | reported |
| Risk Classification | CNN (Final) | Recall | reported |
| Risk Classification | CNN (Final) | F1 Score | reported |

> Precision / Recall / F1 values are available in the notebook output (`down-syndrome-class.ipynb`).

---

## Project Structure

```
down-syndrome-detection/
│
├── down-syndrome-yolo.ipynb       # Stage 1: NT detection (YOLOv8)
├── down-syndrome-class.ipynb      # Stage 2: Risk classification (CNN)
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
│   ├── imgsz640/weights/best.pt   # Phase 1 weights
│   └── imgsz1280/weights/best.pt  # Phase 2 weights (use this for inference)
│
└── README.md
```

---

## Setup & Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/down-syndrome-detection.git
cd down-syndrome-detection

# Install dependencies
pip install ultralytics --upgrade
pip install tensorflow opencv-python matplotlib scikit-learn seaborn tqdm
```

---

## Usage

### Stage 1 — Train YOLOv8 NT Detector

**Phase 1:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    optimizer='SGD',
    lr0=0.001,
    patience=30,
    project="NT-Detection",
    name="imgsz640"
)
```

**Phase 2:**
```python
model = YOLO("NT-Detection/imgsz640/weights/best.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=1280,
    batch=8,
    optimizer='SGD',
    lr0=0.001,
    patience=30,
    project="NT-Detection",
    name="imgsz1280"
)
```

### Stage 2 — Train CNN Classifier

Run all cells in `down-syndrome-class.ipynb`. The final model configuration uses:
- `epochs=150`, `batch_size=32`
- EarlyStopping + ReduceLROnPlateau callbacks
- Adam `lr=1e-3`, L2 `1e-4`, BatchNorm throughout

---

## Demo / Inference

### NT Region Detection

```python
from ultralytics import YOLO
import cv2

model = YOLO("NT-Detection/imgsz1280/weights/best.pt")
results = model.predict(source="path/to/ultrasound.png", imgsz=1280, conf=0.25)

for r in results:
    img = r.plot()
    cv2.imshow("NT Detection", img)
    cv2.waitKey(0)
```

### Down Syndrome Risk Classification

```python
import tensorflow as tf
import cv2
import numpy as np

# Load saved classifier
model = tf.keras.models.load_model("cnn_classifier.h5")

# Preprocess image
img = cv2.imread("path/to/nt_crop.png", cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img = clahe.apply(img)
img = cv2.GaussianBlur(img, (3,3), 0)
img = cv2.resize(img, (512, 512))
img = cv2.merge([img, img, img])
img = np.expand_dims(img, axis=0)

# Predict
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

## Citation

If you use this work, please cite:

```bibtex
@misc{leelab2025downsyndrome,
  author    = {Leela Bhargavi N R},
  title     = {Prenatal Down Syndrome Detection via Nuchal Translucency Analysis in Fetal Ultrasound},
  year      = {2025},
  note      = {MTech Data Science Project, Amrita Vishwa Vidyapeetham},
  url       = {https://github.com/<your-username>/down-syndrome-detection}
}
```

---

## Author

**Leela Bhargavi N R**
MTech Data Science — Amrita Vishwa Vidyapeetham
[LinkedIn](https://www.linkedin.com/in/leela-bhargavi-n-r/) | leelab1113@gmail.com
