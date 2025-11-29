<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
</p>

<h1 align="center">ML Analysis Module</h1>

<p align="center">
  <strong>AutoEncoder-Based Anomaly Detection for Thermal Transformer Images</strong>
</p>

<p align="center">
  Deep learning module using Convolutional AutoEncoders for detecting anomalies in thermal images via reconstruction error analysis.
</p>

---

## Team Arbitary

| Name |
|------|
| **Yasiru Basnayake** |
| **Kumal Loneth** |
| **Dasuni Dissanayake** |
| **Hasitha Gallella** |

---

## Overview

This module implements an **unsupervised anomaly detection** system using AutoEncoders. The model learns to reconstruct "normal" thermal images, and anomalies are detected when the reconstruction error exceeds a threshold.

### Why AutoEncoders?

| Advantage | Description |
|-----------|-------------|
| **Unsupervised Learning** | Only needs normal images for training |
| **Novelty Detection** | Can detect previously unseen failure patterns |
| **Pixel-Level Precision** | Generates localized anomaly maps |
| **Interpretability** | Visual heatmaps explain detections |

---

## Architecture

### AnomalyAutoEncoder

```python
class AnomalyAutoEncoder(nn.Module):
    """
    Input: (batch, 3, 256, 256)
    Latent: (batch, 128, 8, 8)
    Output: (batch, 3, 256, 256)
    """
```

#### Encoder

| Layer | Input | Output | Kernel | Stride |
|-------|-------|--------|--------|--------|
| Conv2d + BN + ReLU | 3x256x256 | 32x128x128 | 3x3 | 2 |
| Conv2d + BN + ReLU | 32x128x128 | 64x64x64 | 3x3 | 2 |
| Conv2d + BN + ReLU | 64x64x64 | 128x32x32 | 3x3 | 2 |
| Conv2d + BN + ReLU | 128x32x32 | 256x16x16 | 3x3 | 2 |
| Conv2d + BN + ReLU | 256x16x16 | 128x8x8 | 3x3 | 2 |

#### Decoder

| Layer | Input | Output | Kernel | Stride |
|-------|-------|--------|--------|--------|
| ConvTranspose2d + BN + ReLU | 128x8x8 | 256x16x16 | 3x3 | 2 |
| ConvTranspose2d + BN + ReLU | 256x16x16 | 128x32x32 | 3x3 | 2 |
| ConvTranspose2d + BN + ReLU | 128x32x32 | 64x64x64 | 3x3 | 2 |
| ConvTranspose2d + BN + ReLU | 64x64x64 | 32x128x128 | 3x3 | 2 |
| ConvTranspose2d + Sigmoid | 32x128x128 | 3x256x256 | 3x3 | 2 |

### ImprovedAutoEncoder (U-Net Style)

An alternative architecture with **skip connections** for better reconstruction:

```
Encoder                    Decoder
   |                          ^
   +--------------------------|  Skip Connection
   |                          |
   +--------------------------|  Skip Connection
   |                          |
   +--------------------------|  Skip Connection
   |                          |
   +---> Bottleneck ----------+
```

---

## Module Structure

```
ML_analysis/
├── model.py               # AutoEncoder architecture
├── train.py               # Initial training script
├── finetune.py            # Finetuning on new data
├── detect_and_annotate.py # Inference & visualization
├── dataset.py             # Dataset loading utilities
├── finetune_dataset.py    # Finetuning dataset loader
├── requirements.txt       # Dependencies
└── models/
    └── best_model.pth     # Trained weights
```

---

## Usage

### Training

```bash
python train.py \
  --data-dir ../Dataset/T1/normal \
  --output-dir models \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.001 \
  --img-size 256
```

**Training Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | Required | Path to normal images |
| `--output-dir` | `models` | Output directory |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--img-size` | `256` | Input image size |
| `--val-split` | `0.2` | Validation split |

### Finetuning

```bash
python finetune.py \
  --feedback-data ../Finetune_data/Local_Dataset \
  --weights models/best_model.pth \
  --output-dir ../Finetune_data/output \
  --epochs 20 \
  --lr 0.0001
```

**Key Differences from Training:**

| Aspect | Training | Finetuning |
|--------|----------|------------|
| Learning Rate | 0.001 | 0.0001 |
| Epochs | 50 | 20 |
| Initialization | Random | Pre-trained weights |
| Purpose | Learn patterns | Adapt to new data |

### Detection

```bash
python detect_and_annotate.py \
  --image ../Dataset/T1/faulty/image.jpg \
  --model models/best_model.pth \
  --threshold 0.5 \
  --min-area 200 \
  --max-area 5000 \
  --max-annotations 3 \
  --output-dir results
```

---

## Detection Pipeline

### Step-by-Step Process

```python
# 1. Load and preprocess image
image_tensor, original_bgr, cropped_rgb, size, coords = preprocess_image(path)

# 2. Generate anomaly map
anomaly_map, reconstructed = model.get_anomaly_map(image_tensor)

# 3. Create binary mask
mask = generate_anomaly_mask(anomaly_map, threshold=0.5)

# 4. Find contours and draw boxes
annotated, boxes = find_contours_and_draw_boxes(
    original_bgr, mask, size, coords,
    min_area=200, max_area=5000, max_annotations=3
)
```

### Preprocessing

1. **Load Image**: Read with OpenCV (BGR format)
2. **Crop Border**: Remove 10% border (reduce edge artifacts)
3. **Resize**: Scale to 256x256 pixels
4. **Normalize**: Scale to [0, 1] range
5. **Convert**: NumPy to PyTorch tensor

### Anomaly Map Generation

```python
def get_anomaly_map(self, x):
    """Generate pixel-wise anomaly scores"""
    with torch.no_grad():
        reconstructed = self.forward(x)
        # Reconstruction error per pixel
        anomaly_map = torch.abs(x - reconstructed)
        # Average across RGB channels
        anomaly_map = torch.mean(anomaly_map, dim=1, keepdim=True)
    return anomaly_map, reconstructed
```

### Contour Detection

1. **Threshold**: Binary mask from anomaly map
2. **Find Contours**: OpenCV `findContours()`
3. **Filter by Area**: Remove too small/large regions
4. **Bounding Boxes**: `cv2.boundingRect()` for each contour
5. **Score**: Mean anomaly value in region
6. **Blue Filter**: Skip regions with high blue content (cold areas)
7. **Top-K**: Keep highest scoring detections

---

## Training Details

### Data Augmentation

```python
transforms.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3
    ),
])
```

### Loss Function

```python
loss = F.mse_loss(reconstructed, original)
```

**MSE (Mean Squared Error)** measures pixel-wise reconstruction quality.

### Early Stopping

```python
patience = 10  # epochs without improvement
```

Training stops if validation loss doesn't improve for 10 consecutive epochs.

### Checkpointing

```python
if val_loss < best_val_loss:
    torch.save(model.state_dict(), 'best_model.pth')
    best_val_loss = val_loss
```

---

## Model Performance

### Training Metrics

| Metric | Value |
|--------|-------|
| Final Training Loss | ~0.005 |
| Final Validation Loss | ~0.008 |
| Convergence | ~30 epochs |

### Detection Metrics

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.3 | 0.75 | 0.95 | 0.84 |
| 0.5 | 0.88 | 0.85 | 0.86 |
| 0.7 | 0.95 | 0.70 | 0.81 |

**Recommended threshold: 0.5** (balanced precision/recall)

---

## Configuration

### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Anomaly map threshold (0-1) |
| `min_area` | 200 | Minimum contour area (pixels) |
| `max_area` | 5000 | Maximum contour area (pixels) |
| `max_annotations` | 3 | Maximum detections to show |
| `blue_threshold` | 30 | Max blue percentage (%) |

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Input Size | 256x256x3 |
| Latent Dimension | 128 |
| Encoder Channels | [32, 64, 128, 256, 128] |
| Activation | ReLU (encoder), Sigmoid (output) |
| Normalization | BatchNorm2d |

---

## API Integration

### Using the Model

```python
from model import AnomalyAutoEncoder
from detect_and_annotate import (
    load_model, preprocess_image, 
    generate_anomaly_mask, find_contours_and_draw_boxes
)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('models/best_model.pth', device)

# Process image
tensor, bgr, rgb, size, coords = preprocess_image('image.jpg')
tensor = tensor.to(device)

# Get anomaly map
anomaly_map, reconstructed = model.get_anomaly_map(tensor)
anomaly_np = anomaly_map.cpu().squeeze().numpy()

# Detect anomalies
mask = generate_anomaly_mask(anomaly_np, threshold=0.5)
annotated, boxes = find_contours_and_draw_boxes(
    bgr, mask, size, coords,
    min_area=200, max_area=5000, max_annotations=3
)

# boxes = [{'id': 1, 'bbox': (x, y, w, h), 'score': 85.5, ...}, ...]
```

---

## Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.21.0
albumentations>=1.3.0
matplotlib>=3.5.0
Pillow>=10.0.0
tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License.

---

<p align="center">
  Made by <strong>Team Arbitary</strong>
</p>
