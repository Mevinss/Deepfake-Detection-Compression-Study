# Deepfake Detection — Web Application & Compression Study

> **Complete deepfake detection web application with video upload and real-time analysis.**  
> **Полноценное веб-приложение для детекции дипфейков с загрузкой видео и анализом в реальном времени.**

🌐 **[Русская версия документации](README_RU.md)** | 🚀 **[Quick Start (RU)](QUICKSTART_RU.md)**

This project provides a web interface for deepfake detection in videos. It benchmarks three lightweight CNN architectures — **MobileNetV3-Large**, **EfficientNet-B0**, and **GhostNet** — for their ability to detect deepfake faces after H.264 video compression at varying Constant Rate Factor (CRF) levels (23, 32, 40).

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/Mevinss/Deepfake-Detection-Compression-Study.git
cd Deepfake-Detection-Compression-Study
pip install -r requirements.txt

# Run web application
python app.py

# Open browser: http://127.0.0.1:5000
```

**📖 For detailed instructions in Russian, see [QUICKSTART_RU.md](QUICKSTART_RU.md)**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Robustness Evaluation](#robustness-evaluation)
- [Results](#results)

---

## Project Overview

| Component | Description |
|-----------|-------------|
| **Data preprocessing** | OpenCV frame extraction, MediaPipe / MTCNN face detection & crop, albumentations-based compression simulation |
| **Models** | MobileNetV3-Large, EfficientNet-B0, GhostNet (via `timm`) with optional CBAM attention blocks |
| **Training** | PyTorch Lightning *or* pure PyTorch, BCEWithLogitsLoss, AdamW + CosineAnnealingLR, TensorBoard logging |
| **Evaluation** | Accuracy, F1-score, ROC-AUC across CRF levels 0 / 23 / 32 / 40, auto-generated plots |

---

## Repository Structure

```
Deepfake-Detection-Compression-Study/
├── configs/
│   └── config.yaml          # Training & evaluation configuration
├── src/
│   ├── data/
│   │   ├── preprocess.py    # Frame extraction, face detection, augmentation
│   │   └── dataset.py       # PyTorch Dataset (real / fake directories)
│   ├── models/
│   │   ├── attention.py     # ChannelAttention, SpatialAttention, CBAM
│   │   └── classifier.py    # build_model() factory for all three backbones
│   ├── train.py             # Training pipeline (Lightning + pure PyTorch)
│   └── evaluate.py          # Robustness evaluation & plot generation
├── requirements.txt
└── README.md
```

> **Data directories** (not tracked by Git):
> ```
> data/
>   train/  real/  fake/
>   val/    real/  fake/
>   test/   real/  fake/
> ```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mevinss/Deepfake-Detection-Compression-Study.git
cd Deepfake-Detection-Compression-Study

# 2. Create a virtual environment (Python 3.9+)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

> **Note:** `ffmpeg` must be installed on the system PATH for the video
> compression helper in `src/data/preprocess.py`.

---

## Data Preparation

### 1. Extract frames from videos

```python
from src.data.preprocess import extract_frames

paths = extract_frames(
    video_path="videos/fake_001.mp4",
    output_dir="frames/fake_001",
    max_frames=300,
    frame_interval=5,   # extract 1 frame every 5
)
```

### 2. Detect and crop faces

```python
from src.data.preprocess import get_face_detector, crop_faces
import cv2

detector = get_face_detector("mediapipe")   # or "mtcnn"
image = cv2.imread("frames/fake_001/frame_000000.png")
crops = crop_faces(image, detector, target_size=(224, 224), margin=0.2)
```

### 3. Simulate video compression (offline)

```python
from src.data.preprocess import compress_video_ffmpeg

compress_video_ffmpeg("input.mp4", "input_crf32.mp4", crf=32)
```

Place the resulting face crops under `data/train/real/`, `data/train/fake/`,
`data/val/`, and `data/test/` following the expected directory layout.

---

## Configuration

All hyper-parameters are controlled via `configs/config.yaml`:

```yaml
model:
  name:          mobilenetv3    # mobilenetv3 | efficientnet_b0 | ghostnet
  pretrained:    true
  use_attention: true           # attach CBAM attention block
  hidden_dim:    256
  dropout:       0.3

image_size:  224
batch_size:  32
epochs:      20
lr:          0.0001

crf_levels: [0, 23, 32, 40]    # CRF=0 = no compression (baseline)
eval_models:
  - mobilenetv3
  - efficientnet_b0
  - ghostnet
```

---

## Training

### PyTorch Lightning (recommended)

```bash
python src/train.py --config configs/config.yaml
```

Monitor training in TensorBoard:

```bash
tensorboard --logdir runs/
```

### Pure PyTorch (no Lightning required)

```bash
python src/train.py --config configs/config.yaml --no_lightning
```

Checkpoints are saved to `checkpoints/<model_name>_best.pth`.

---

## Robustness Evaluation

After training all three models, run the evaluation script:

```bash
python src/evaluate.py --config configs/config.yaml --output_dir results/
```

This will:
1. Load each checkpoint from `checkpoints/`.
2. Evaluate on the test set with simulated JPEG compression for each CRF level.
3. Save metric plots to `results/robustness_acc.png`, `results/robustness_f1.png`,
   `results/robustness_auc.png`.
4. Dump raw numbers to `results/robustness_results.json`.

---

## Results

After evaluation, plots are saved to the `results/` directory.  
Example shape of the generated figure for Accuracy vs CRF:

```
Accuracy
  1.0 ┤ ───●── MobileNetV3
  0.9 ┤   ╱ ──●── EfficientNet-B0
  0.8 ┤  ╱  ╱  ──●── GhostNet
  0.7 ┤ ...
      └──────────────────── CRF
       40   32   23    0
```

*(Higher CRF = more compression = harder task.)*

