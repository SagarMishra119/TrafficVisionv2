# Traffic Density Detection and Congestion Analysis

> **YOLOv11-based modular deep learning pipeline** for real-time vehicle detection, counting, traffic density classification, and congestion analysis. Production-ready. API-first. UI-integration-ready.

---

## Dataset Recommendation

**Primary: Vehicle Detection at Any Angle (Roboflow)**
| Attribute | Detail |
|---|---|
| Images | ~3,000+ annotated frames |
| Classes | car, bus, truck, motorcycle, bicycle |
| Format | YOLO-native (no conversion needed) |
| Splits | Train / Valid / Test pre-split |
| Expected mAP@0.5 | 85–92% (YOLOv8s) |

**Why this dataset?**
- All 5 required vehicle classes are annotated
- Real-world traffic scenes (highways, intersections, urban roads)
- High annotation quality with varied angles and occlusions
- Direct YOLOv8 compatibility — zero preprocessing overhead
- Feasible training time: ~2–3 hours on a single GPU (RTX 3060+)

**Alternative: UA-DETRAC** — 140k+ frames, highway/urban, strong benchmark baselines.

---

## Project Structure

```
traffic_density_detection/
├── dataset/
│   ├── train/images/  ├── train/labels/
│   ├── valid/images/  ├── valid/labels/
│   └── test/images/   └── test/labels/
│
├── configs/
│   ├── settings.yaml          # Master config (thresholds, training, paths)
│   └── dataset.yaml           # YOLO dataset config (auto-generated)
│
├── models/
│   ├── weights/               # Saved .pt files
│   └── exports/               # ONNX / TorchScript exports
│
├── notebooks/                 # EDA and experimentation
├── reports/
│   ├── plots/                 # All generated charts
│   └── *.json                 # Metrics, comparisons, pipeline summary
│
├── scripts/
│   └── download_dataset.py    # Dataset download & setup
│
├── src/
│   ├── data/
│   │   ├── dataset_manager.py     # Download, organize, split
│   │   ├── dataset_validator.py   # Corrupt/missing/duplicate checks
│   │   ├── augmentation.py        # Albumentations augmentation pipeline
│   │   └── dataset_stats.py       # Statistics + visualizations
│   │
│   ├── training/
│   │   ├── trainer.py             # YOLOv8n / YOLOv8s trainer
│   │   └── train_pipeline.py      # End-to-end training orchestrator
│   │
│   ├── evaluation/
│   │   ├── evaluator.py           # mAP, precision, recall, F1
│   │   └── metrics.py             # Loss curves, confusion matrix, PR curves
│   │
│   ├── detection/
│   │   ├── detector.py            # Image / video / webcam / folder detection
│   │   └── inference.py           # Full inference pipeline
│   │
│   ├── counting/
│   │   └── counter.py             # Per-class vehicle counting
│   │
│   ├── density/
│   │   └── classifier.py          # Low / Medium / High classification
│   │
│   ├── analytics/
│   │   ├── analytics.py           # Session logging, summaries, export
│   │   └── visualizer.py          # Traffic dashboards and plots
│   │
│   ├── api/
│   │   ├── app.py                 # FastAPI application + CORS
│   │   ├── routes.py              # All API endpoints
│   │   └── schemas.py             # Pydantic request/response models
│   │
│   └── utils/
│       ├── config.py              # YAML config loader
│       ├── file_utils.py          # File I/O helpers
│       ├── image_utils.py         # OpenCV wrappers
│       └── logger.py              # Loguru logger setup
│
├── tests/
│   └── test_modules.py            # Unit tests (unittest)
│
├── main.py                        # CLI entry point
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install Dependencies

```bash
git clone <repo>
cd traffic_density_detection
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Option A: Roboflow (recommended)
python scripts/download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

# Option B: Synthetic data (no internet, for testing)
python scripts/download_dataset.py --synthetic
```

### 3. Validate Dataset

```bash
python main.py train --step validate
```

### 4. Generate Dataset Statistics

```bash
python main.py train --step stats
```

### 5. Train YOLOv8n (Baseline)

```bash
python main.py train --step nano --dataset-yaml configs/dataset.yaml
```

### 6. Train YOLOv8s (Improved)

```bash
python main.py train --step small --dataset-yaml configs/dataset.yaml
```

### 7. Run Full Pipeline (All Steps)

```bash
python main.py train --step all --dataset-yaml configs/dataset.yaml --export-format onnx
```

### 8. Evaluate a Model

```bash
python main.py evaluate --weights models/weights/yolov8s_traffic_best.pt --split test
```

---

## Inference Commands

```bash
# Single image
python main.py infer --mode image --source path/to/image.jpg \
    --weights models/weights/yolov8s_traffic_best.pt

# Folder of images
python main.py infer --mode folder --source path/to/images/ \
    --weights models/weights/yolov8s_traffic_best.pt \
    --output-dir reports/inference

# Video file
python main.py infer --mode video --source traffic.mp4 \
    --weights models/weights/yolov8s_traffic_best.pt \
    --output-video reports/inference/output.mp4

# Live webcam
python main.py infer --mode webcam \
    --weights models/weights/yolov8s_traffic_best.pt \
    --camera 0
```

---

## API Server

```bash
python main.py api --weights models/weights/yolov8s_traffic_best.pt --port 8000
```

**Swagger UI:** http://localhost:8000/docs

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Server + model status |
| `/api/v1/predict-image` | POST | Detect vehicles in uploaded image |
| `/api/v1/predict-video` | POST | Process uploaded video |
| `/api/v1/get-density` | GET | Classify density by vehicle count |
| `/api/v1/update-density-thresholds` | POST | Change Low/Medium/High thresholds |
| `/api/v1/get-analytics` | GET | Session analytics summary |
| `/api/v1/clear-analytics` | DELETE | Reset session data |

**Example API call:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict-image?return_image=true" \
     -F "file=@traffic.jpg"
```

**Response:**
```json
{
  "status": "success",
  "source": "traffic.jpg",
  "detections": [
    {"vehicle_type": "car", "confidence": 0.91, "bbox": [120, 80, 320, 240], "class_id": 0}
  ],
  "counts": {
    "total_vehicles": 12,
    "cars": 8,
    "buses": 1,
    "trucks": 2,
    "motorcycles": 1,
    "bicycles": 0
  },
  "density": {
    "vehicle_count": 12,
    "density": "Medium",
    "density_level": "medium",
    "color": "#FFD600",
    "threshold_range": "11-25"
  },
  "congestion_index": 47.2
}
```

---

## Run Unit Tests

```bash
python main.py test
# or
python -m pytest tests/ -v
```

---

## Configuration

Edit `configs/settings.yaml` to tune:

```yaml
density_thresholds:
  low:    { min: 0,  max: 10, label: Low,    color: "#00C853" }
  medium: { min: 11, max: 25, label: Medium, color: "#FFD600" }
  high:   { min: 26, max: 999, label: High,  color: "#D50000" }

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.01
  patience: 20   # early stopping
```

---

## Expected Training Results

| Model | mAP@0.5 | mAP@0.5:0.95 | FPS (RTX 3060) | Size |
|---|---|---|---|---|
| YOLOv8n | ~82–86% | ~55–60% | ~120 FPS | 6 MB |
| YOLOv8s | ~87–92% | ~62–68% | ~90 FPS | 22 MB |

---

## UI Integration

The API is designed to connect directly to:

- **Streamlit** — `requests.post("/api/v1/predict-image", files={"file": ...})`
- **React / Next.js** — `fetch("/api/v1/predict-image", { method: "POST", body: formData })`
- **FastAPI Dashboard** — mount this service as a microservice behind a proxy
- **Grafana / BI Tools** — consume `/get-analytics` for real-time dashboards

No UI code lives inside ML modules. All outputs are structured JSON.

---

## Deployment

### Docker (recommended for production)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main.py", "api", "--weights", "models/weights/yolov8s_traffic_best.pt"]
```

```bash
docker build -t traffic-detection .
docker run -p 8000:8000 -v $(pwd)/models:/app/models traffic-detection
```

### Systemd Service

```ini
[Unit]
Description=Traffic Detection API

[Service]
WorkingDirectory=/opt/traffic_density_detection
ExecStart=/usr/bin/python main.py api --weights models/weights/yolov8s_traffic_best.pt
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Augmentation | Albumentations |
| API Framework | FastAPI + Uvicorn |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Logging | Loguru |
| Testing | unittest / pytest |
| Export | ONNX, TorchScript |

---

*Built as a production-ready internship-level ML project. Designed for seamless integration with Streamlit, React, Next.js, and dashboard applications.*
