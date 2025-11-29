<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
</p>

<h1 align="center">Arbit Transformer Analysis</h1>

<p align="center">
  <strong>AI-Powered Thermal Anomaly Detection System for Electrical Transformers</strong>
</p>

<p align="center">
  An advanced machine learning system that combines AutoEncoder-based anomaly detection with thermal hotspot analysis to identify potential faults in transformer thermal images.
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API</a> •
  <a href="#team">Team</a>
</p>

---

## Team Arbitary

| Name | Role |
|------|------|
| **Yasiru Basnayake** | Full Stack Developer |
| **Kumal Loneth** | Backend Developer |
| **Dasuni Dissanayake** | Frontend Developer |
| **Hasitha Gallella** | ML/AI Engineer |

---

## Features

- **Dual Detection System** - Combines ML and thermal analysis for comprehensive detection
- **AutoEncoder Model** - Deep learning-based anomaly detection via reconstruction error
- **Thermal Hotspot Detection** - Region-based temperature analysis using DBSCAN clustering
- **Unified Analysis** - Merges results from both methods with annotated visualizations
- **REST API** - FastAPI-based service for integration with other systems
- **Auto-Finetuning** - Automated model improvement with user feedback
- **Confidence Scoring** - Severity classification with confidence metrics

---

## Architecture

### ML Pipeline Overview

```
+-----------------------------------------------------------------------+
|                        UNIFIED THERMAL ANALYSIS                        |
+-----------------------------------------------------------------------+
                                    |
                    +---------------+---------------+
                    v                               v
        +-------------------+           +-------------------+
        |   ML AutoEncoder  |           | Thermal Detector  |
        |     Analysis      |           |     Analysis      |
        +---------+---------+           +---------+---------+
                  |                               |
                  v                               v
        +-------------------+           +-------------------+
        | Reconstruction    |           | Red Channel       |
        | Error Map         |           | Hotspot Map       |
        +---------+---------+           +---------+---------+
                  |                               |
                  v                               v
        +-------------------+           +-------------------+
        | Contour Detection |           | DBSCAN Clustering |
        | & Bounding Boxes  |           | & Bounding Boxes  |
        +---------+---------+           +---------+---------+
                  |                               |
                  +---------------+---------------+
                                  v
                    +-----------------------+
                    |   Combined Output     |
                    |  ML (RED) + TH (YELLOW)|
                    +-----------------------+
```

### AutoEncoder Architecture

```
Input Image (256x256x3)
        |
+----------------------------------------+
|              ENCODER                   |
+----------------------------------------+
| Conv2d(3->32) + BN + ReLU + Stride(2)  | -> 128x128x32
| Conv2d(32->64) + BN + ReLU + Stride(2) | -> 64x64x64
| Conv2d(64->128) + BN + ReLU + Stride(2)| -> 32x32x128
| Conv2d(128->256) + BN + ReLU + Stride(2)| -> 16x16x256
| Conv2d(256->128) + BN + ReLU + Stride(2)| -> 8x8x128
+----------------------------------------+
        |
   Latent Space (8x8x128)
        |
+----------------------------------------+
|              DECODER                   |
+----------------------------------------+
| ConvTranspose2d(128->256) + BN + ReLU  | -> 16x16x256
| ConvTranspose2d(256->128) + BN + ReLU  | -> 32x32x128
| ConvTranspose2d(128->64) + BN + ReLU   | -> 64x64x64
| ConvTranspose2d(64->32) + BN + ReLU    | -> 128x128x32
| ConvTranspose2d(32->3) + Sigmoid       | -> 256x256x3
+----------------------------------------+
        |
Reconstructed Image (256x256x3)
```

### How Anomaly Detection Works

| Phase | Process | Output |
|-------|---------|--------|
| **Training** | Model learns to reconstruct normal thermal images | Low reconstruction error on normal patterns |
| **Inference** | New image fed through AutoEncoder | Reconstruction + Error Map |
| **Detection** | High error regions indicate anomalies | Bounding boxes with confidence scores |

**Key Insight**: Anomalies produce high reconstruction error because the model hasn't learned to reconstruct abnormal patterns.

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **CUDA** (optional, for GPU acceleration)
- **Docker** (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/Team-Arbitary/Arbit-Transformer-Analysis.git
cd Arbit-Transformer-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements-api.txt
```

### Running the API

```bash
# Development mode
python api.py

# Production mode with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000
```

**API available at:** `http://localhost:8000`

### Docker Deployment

```bash
# Build image
docker build -t arbit-analysis .

# Run container
docker run -p 8000:8000 arbit-analysis
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API documentation |
| `GET` | `/health` | Health check status |
| `GET` | `/config` | Current configuration |
| `POST` | `/detect` | Analyze thermal image |

### Detection Endpoint

```http
POST /detect
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `baseline` | File | Yes | Baseline thermal image (accepted but not used) |
| `maintenance` | File | Yes | Thermal image to analyze |
| `transformer_id` | String | Yes | Transformer identifier |
| `return_format` | String | No | Response format: `json`, `annotated`, `complete` |

### Response Format

```json
{
  "status": "success",
  "transformer_id": "TRF-001",
  "summary": {
    "total_anomalies": 3,
    "severity_distribution": {
      "HIGH": 1,
      "MEDIUM": 1,
      "LOW": 1,
      "MINIMAL": 0
    },
    "total_anomaly_area": 2500,
    "average_confidence": 0.85,
    "critical_anomalies": 1,
    "detection_quality": "HIGH"
  },
  "anomalies": [
    {
      "id": 1,
      "bbox": [120, 80, 50, 40],
      "center": [145, 100],
      "area": 2000,
      "avg_temp_change": 145.5,
      "max_temp_change": 165.5,
      "severity": 0.85,
      "type": "heating",
      "confidence": 1.0,
      "reasoning": "Significant temperature increase detected...",
      "severity_level": "HIGH",
      "severity_color": [0, 0, 255]
    }
  ],
  "detection_methods": ["statistical", "computer_vision"],
  "annotated_image_base64": "data:image/png;base64,..."
}
```

### Severity Classification

| Severity | Score Range | Color | Description |
|----------|-------------|-------|-------------|
| **HIGH** | >= 0.8 | Red | Critical - Immediate attention required |
| **MEDIUM** | 0.6 - 0.8 | Orange | Warning - Schedule maintenance |
| **LOW** | 0.4 - 0.6 | Yellow | Monitor - Track for changes |
| **MINIMAL** | < 0.4 | Green | Normal - No action needed |

---

## Auto-Finetuning System

### Workflow

```
+-------------------------------------------------------------+
|                    AUTO-FINETUNE WORKFLOW                   |
+-------------------------------------------------------------+

1. User Feedback Collection
   --> Images saved to temp_data/normal & temp_data/faulty

2. Threshold Check (min_images = 6)
   --> If normal images >= threshold
       |--> Move to Local_Dataset/MM_YYYY/
       +--> Trigger finetuning

3. Finetuning Process
   --> python ML_analysis/finetune.py
       |--> Load best_model.pth
       |--> Train on new data (lr=0.0001, epochs=20)
       +--> Save updated model

4. Cleanup
   |--> Success: Clear temp_data
   +--> Failure: Rollback new dataset folder

5. Status Logging
   --> Write finetune_status.json
```

### Running Auto-Finetune

```bash
python auto_finetune.py \
  --temp-data Finetune_data/temp_data \
  --local-dataset Finetune_data/Local_Dataset \
  --min-images 6 \
  --finetune-script ML_analysis/finetune.py
```

---

## Project Structure

```
Arbit-Transformer-Analysis/
├── api.py                      # FastAPI service
├── auto_api.py                 # API with auto-finetune integration
├── auto_finetune.py            # Automated finetuning workflow
├── unified_thermal_analysis.py # Combined analysis pipeline
├── config.yaml                 # Configuration file
├── requirements.txt            # Core dependencies
├── requirements-api.txt        # API dependencies
├── Dockerfile                  # Docker configuration
├── render.yaml                 # Render deployment config
│
├── ML_analysis/                # Machine Learning module
│   ├── model.py               # AutoEncoder architecture
│   ├── train.py               # Initial training script
│   ├── finetune.py            # Finetuning script
│   ├── detect_and_annotate.py # Detection & visualization
│   ├── dataset.py             # Dataset utilities
│   ├── finetune_dataset.py    # Finetune dataset loader
│   └── models/
│       └── best_model.pth     # Trained model weights
│
├── heat_point_analysis/        # Thermal analysis module
│   └── thermal_hotpoint_detector.py
│
├── Dataset/                    # Training data
│   └── T1/
│       ├── normal/
│       └── faulty/
│
├── Finetune_data/             # Finetuning data
│   ├── temp_data/
│   │   ├── normal/
│   │   └── faulty/
│   ├── Local_Dataset/
│   └── output/
│
└── unified_results/           # Analysis output
```

---

## Configuration

### config.yaml

```yaml
detection:
  statistical:
    temperature_threshold: 6.0    # Std deviations
    min_anomaly_size: 200         # Min pixels

  thermal:
    hot_spot_threshold: 0.75
    temperature_threshold: 200
    min_cluster_size: 15
    epsilon: 20                   # DBSCAN epsilon

  ml:
    threshold: 0.5                # Anomaly map threshold
    min_area: 200                 # Min contour area
    max_area: 5000                # Max contour area
    max_annotations: 3            # Top N detections
    blue_threshold: 30            # Blue filter %

  confidence:
    min_confidence: 0.6

model:
  path: "ML_analysis/models/best_model.pth"

visualization:
  output:
    save_results: true
    result_format: "png"
```

---

## Training the Model

### Initial Training

```bash
cd ML_analysis

python train.py \
  --data-dir ../Dataset/T1/normal \
  --output-dir models \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.001
```

### Finetuning

```bash
python finetune.py \
  --feedback-data ../Finetune_data/Local_Dataset \
  --weights models/best_model.pth \
  --output-dir ../Finetune_data/output \
  --epochs 20 \
  --lr 0.0001
```

---

## Command Line Tools

### Unified Analysis

```bash
python unified_thermal_analysis.py Dataset/T1/faulty/image.jpg \
  --threshold 0.5 \
  --min-area 200 \
  --max-area 5000 \
  --max-annotations 3 \
  --thermal-threshold 200 \
  --output-dir unified_results
```

### Output Files

- `{filename}_unified_analysis.png` - Full visualization grid
- `{filename}_combined_annotated.jpg` - Combined annotation image
- `{filename}_unified_report.txt` - Text report

---

## Docker Deployment

### Build and Run

```bash
# Build
docker build -t arbit-analysis .

# Run
docker run -d \
  --name arbit-analysis \
  -p 8000:8000 \
  -v $(pwd)/ML_analysis/models:/app/ML_analysis/models \
  arbit-analysis
```

### Docker Compose

```yaml
version: '3.8'
services:
  ml-analysis:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./ML_analysis/models:/app/ML_analysis/models
    environment:
      - PORT=8000
    restart: unless-stopped
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Inference Time** | ~200ms/image (CPU), ~50ms (GPU) |
| **Model Size** | ~15MB |
| **Input Resolution** | 256x256 pixels |
| **Detection Accuracy** | ~92% (validation set) |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made by <strong>Team Arbitary</strong>
</p>
