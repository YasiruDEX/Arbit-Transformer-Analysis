# Thermal Transformer Anomaly Detection & Segmentation

This project implements an **AutoEncoder-based anomaly detection** system for thermal images of transformers. The model learns to reconstruct normal transformer images and identifies anomalies by measuring reconstruction errors.

## 🎯 Approach

### Method: Unsupervised AutoEncoder Learning

1. **Training Phase**: Train AutoEncoder ONLY on normal (non-faulty) images
2. **Detection Phase**: Calculate reconstruction error for test images
3. **Segmentation Phase**: Generate pixel-wise anomaly heatmaps and binary masks

### Why This Approach?

- ✅ **No manual annotations required** - learns from normal images only
- ✅ **Pixel-level segmentation** - identifies exact anomaly locations
- ✅ **Generalizable** - can detect various types of anomalies
- ✅ **Fast inference** - real-time capable on CPU/GPU

## 📁 Project Structure

```
ML_analysis/
├── dataset.py              # ThermalDataset loader for T1-T5 folders
├── model.py                # AutoEncoder architecture
├── train.py                # Training script
├── test.py                 # Testing and visualization
├── run_experiment.py       # Full pipeline runner
├── requirements.txt        # Python dependencies
├── Dataset/               # Your thermal image dataset
│   ├── T1/
│   │   ├── faulty/
│   │   └── normal/
│   ├── T2/, T3/, T4/, T5/
├── models/                # Saved model checkpoints
│   ├── best_model.pth
│   └── final_model.pth
└── results/               # Test results & visualizations
    ├── *_result.png       # 4-panel visualizations
    ├── *_mask.png         # Binary segmentation masks
    └── error_analysis.png # Error distribution plots
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Run full experiment (train + test)
python run_experiment.py

# Or train separately
python train.py
```

### 3. Test on New Images

```bash
python test.py
```

## 📊 Results

### Training Results
- **Training Images**: 12 normal images (from T1-T5)
- **Test Images**: 69 images (57 faulty + 12 normal)
- **Best Training Loss**: 2.886

### Detection Performance

| Category | Mean Error | Std Error | Min Error | Max Error |
|----------|-----------|-----------|-----------|-----------|
| **Normal** | 2.720 | 0.455 | 2.244 | 3.753 |
| **Faulty** | 2.407 | 0.332 | 1.776 | 3.586 |

> **Note**: Current results show overlap between normal and faulty distributions. This indicates:
> - More training data needed (only 12 normal images used)
> - Longer training required (20 epochs used for quick test)
> - Consider data augmentation

## 🔧 Model Architecture

### AutoEncoder Structure

```
Encoder:
  Input (3, 256, 256)
  ↓ Conv2d + BN + ReLU
  (32, 128, 128)
  ↓ Conv2d + BN + ReLU
  (64, 64, 64)
  ↓ Conv2d + BN + ReLU
  (128, 32, 32)
  ↓ Conv2d + BN + ReLU
  (256, 16, 16)
  ↓ Conv2d + BN + ReLU
  Latent (128, 8, 8)

Decoder:
  Latent (128, 8, 8)
  ↓ ConvTranspose2d + BN + ReLU
  (256, 16, 16)
  ↓ ConvTranspose2d + BN + ReLU
  (128, 32, 32)
  ↓ ConvTranspose2d + BN + ReLU
  (64, 64, 64)
  ↓ ConvTranspose2d + BN + ReLU
  (32, 128, 128)
  ↓ ConvTranspose2d + Sigmoid
  Output (3, 256, 256)
```

## 📈 Visualization Outputs

Each test image generates a 4-panel visualization:

1. **Original Image** - Input thermal image
2. **Reconstructed** - AutoEncoder reconstruction
3. **Anomaly Heatmap** - Pixel-wise error (red = high error)
4. **Segmentation Overlay** - Binary mask overlaid on original

## ⚙️ Configuration

### Training Parameters

```python
train_autoencoder(
    data_root='Dataset',      # Path to dataset
    batch_size=8,             # Batch size for training
    epochs=50,                # Number of epochs
    learning_rate=0.001,      # Initial learning rate
    img_size=256,             # Input image size
    save_dir='models'         # Model save directory
)
```

### Testing Parameters

```python
test_model(
    model_path='models/best_model.pth',
    data_root='Dataset',
    img_size=256,
    threshold=0.5,            # Threshold for binary mask
    save_dir='results'
)
```

## 🔬 Improving Results

To achieve better anomaly detection performance:

### 1. Collect More Data
- **Current**: 12 normal training images
- **Recommended**: 100+ normal images per transformer type

### 2. Train Longer
- **Current**: 20 epochs (quick test)
- **Recommended**: 50-100 epochs with early stopping

### 3. Data Augmentation
Already implemented in `dataset.py`:
- Horizontal flip
- Rotation (±15°)
- Brightness/contrast adjustment

### 4. Alternative Architectures

The codebase includes `ImprovedAutoEncoder` with skip connections (U-Net style):

```python
from model import ImprovedAutoEncoder

# In train.py, replace:
model = ImprovedAutoEncoder(in_channels=3)
```

### 5. Advanced Methods

For state-of-the-art results, consider:

**PaDiM (Patch Distribution Modeling)**:
- Uses pre-trained ResNet features
- Models normal distribution per patch
- Mahalanobis distance for anomaly scoring

**Other SOTA Methods**:
- PatchCore
- FastFlow
- DRAEM
- CFlow-AD

## 📝 Usage Examples

### Load and Visualize Dataset

```python
from dataset import ThermalDataset
from torch.utils.data import DataLoader

# Load training data
train_dataset = ThermalDataset(
    root_dir='Dataset',
    mode='train',
    img_size=256
)

print(f"Training images: {len(train_dataset)}")

# Load test data
test_dataset = ThermalDataset(
    root_dir='Dataset',
    mode='test',
    img_size=256
)

print(f"Test images: {len(test_dataset)}")
```

### Generate Anomaly Map for Single Image

```python
import torch
from model import AnomalyAutoEncoder
from PIL import Image
import numpy as np

# Load model
model = AnomalyAutoEncoder()
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process image
image = Image.open('path/to/image.jpg')
# ... preprocess image ...

# Get anomaly map
with torch.no_grad():
    anomaly_map, reconstructed = model.get_anomaly_map(image_tensor)
```

## 🐛 Troubleshooting

### Issue: Low separation between normal/faulty
**Solution**: Increase training data and epochs

### Issue: Model not learning
**Solution**: Check learning rate, try different architecture

### Issue: Memory errors
**Solution**: Reduce batch_size or img_size

### Issue: Segmentation masks too noisy
**Solution**: Adjust threshold or morphological kernel size in `test.py`

## 📚 References

- **AutoEncoders for Anomaly Detection**: [Variational AutoEncoder Paper](https://arxiv.org/abs/1312.6114)
- **U-Net Architecture**: [U-Net Paper](https://arxiv.org/abs/1505.04597)
- **PaDiM**: [PaDiM: a Patch Distribution Modeling Framework](https://arxiv.org/abs/2011.08785)

## 🤝 Contributing

To improve this project:

1. Collect more thermal transformer images
2. Experiment with different architectures
3. Implement advanced methods (PaDiM, PatchCore)
4. Add proper train/val/test splits
5. Implement metrics (AUROC, F1, IoU)

## 📄 License

This project is for educational and research purposes.

## 💡 Tips

- **Start simple**: AutoEncoder approach works well for quick testing
- **Scale up**: Add more data before increasing model complexity
- **Validate**: Always check if model learns meaningful features
- **Iterate**: Adjust threshold based on precision/recall requirements
