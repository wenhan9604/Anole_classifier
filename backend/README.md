# Florida Anole Classifier - Backend API

FastAPI backend serving the 3-stage ML pipeline for anole species detection and classification.

## Architecture

### 3-Stage Pipeline

1. **Detection (YOLOv8)**: Detects lizards in uploaded images
2. **Cropping**: Extracts detected regions
3. **Classification (Swin Transformer)**: Classifies cropped images into 5 species

### Models

- **Detection**: Fine-tuned YOLOv8n (`Spring_2025/ultralytics_runs/detect/train_yolov8n_v2/weights/best.pt`)
- **Classification**: Fine-tuned Swin Transformer (Base) - Local: `model_export/swin_transformer_base_lizard_v4/` or HuggingFace: `swin-base-patch4-window12-384-finetuned-lizard-class-swin-base`

## Quick Start

### 1) Create and activate a virtual environment

macOS/Linux:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
cd backend
py -m venv .venv
.venv\\Scripts\\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt

# Install ML dependencies (required for prediction endpoint)
pip install ultralytics transformers torch Pillow
```

### 3) Run the API server

```bash
uvicorn app.main:app --reload --port 8000
```

The server will be available at `http://127.0.0.1:8000` and docs at `http://127.0.0.1:8000/docs`.

## Advanced Setup (Conda)

### Prerequisites

- Conda (Anaconda or Miniconda)
- Python 3.11+ (installed via conda)

### Setup

1. Create conda environment with all dependencies:
```bash
cd backend
conda env create -f environment.yml
conda activate anole-classifier
```

2. To update an existing environment:
```bash
conda env update -f environment.yml --prune
```

### Configuration

3. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env with your model paths if needed
```

4. Ensure models are available:
   - YOLOv8 weights should be at the path specified in `.env`
   - Swin Transformer will be downloaded from HuggingFace on first run

## Running the Server

### Development

```bash
python -m app.main
```

Or with uvicorn directly:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

```bash
# Build image
docker build -t anole-classifier-backend .

# Run container
docker run -p 8000:8000 -v /path/to/models:/app/models anole-classifier-backend
```

## Endpoints

### Core ML Pipeline
- `POST /api/predict` — Upload an image for prediction (multipart: image file)

**Request:**
- `file`: Image file (multipart/form-data)

**Response:**
```json
{
  "totalLizards": 2,
  "predictions": [
    {
      "species": "Green Anole",
      "scientificName": "Anolis carolinensis",
      "confidence": 0.94,
      "count": 1,
      "box": [120.5, 80.2, 250.3, 200.1]
    }
  ]
}
```

**Note**: The `box` field contains bounding box coordinates `[x1, y1, x2, y2]` in the original image coordinate system. Confidence scores are calibrated using temperature scaling (T=2.0) for more realistic values.

- `GET /api/model-info` — Get information about loaded models
- `GET /api/health` — Health check endpoint

### Frontend Support
- `POST /api/observations` — Upload an observation (multipart: image + fields)
- `GET /api/observations` — List observations (mock)
- `GET /api/species/search?q=...` — Search species (mock)
- `GET /api/species/{scientific_name}` — Species details (mock)
- `POST /api/auth/mock-login` — Mock auth tokens

## ML Pipeline Features

### Detection and Classification
The `/api/predict` endpoint uses a 3-stage pipeline:

1. **YOLOv8 Detection**: Detects lizards with bounding boxes
2. **Deduplication**: Merges overlapping detections using IoU and center distance
3. **Swin Classification**: Classifies each unique detection into 5 species

### Confidence Calibration
- **Temperature Scaling**: Default T=2.0 for more realistic confidence scores
- **Advanced Calibration**: Support for external calibrators via `CALIBRATION_PATH` env var
- **Methods**: Temperature scaling, Platt scaling OvR, Isotonic regression OvR

### Model Paths (Auto-detected)
- **YOLO**: `Spring_2025/ultralytics_runs/detect/train_yolov8n_v2/weights/best.pt`
- **Swin**: `model_export/swin_transformer_base_lizard_v4/` (local) or HuggingFace fallback

### Environment Variables (Optional)
```bash
export DETECTION_WEIGHTS_PATH="path/to/your/yolo/weights.pt"
export CLASSIFICATION_MODEL_ID="your-model-id-or-path"
export CALIBRATION_PATH="path/to/calibration/model"
```

### How teammates should place model files locally
- Download YOLOv8 weights and place at:
  - `Spring_2025/ultralytics_runs/detect/train_yolov8n_v2/weights/best.pt`
- Download the Swin model folder (containing `config.json` and model weights) and place at:
  - `model_export/swin_transformer_base_lizard_v4/`
- You can also point to custom locations using env vars before launching the backend:
```bash
export DETECTION_WEIGHTS_PATH="/absolute/path/to/best.pt"
export CLASSIFICATION_MODEL_ID="/absolute/path/to/swin_transformer_base_lizard_v4"  # or a HuggingFace ID
```

## Testing

```bash
# Test with curl
curl -X POST "http://localhost:8000/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/anole_image.jpg"
```

## CORS

By default, CORS allows `http://localhost:5173`. To override, set `CORS_ORIGINS` env var with comma-separated origins.

## Troubleshooting

### Model loading fails
- Verify model paths in `.env`
- Check that YOLO weights exist at specified path
- Ensure HuggingFace model name is correct

## Development

### Project Structure

```
backend/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── models/
│   │   ├── pipeline.py            # 3-stage pipeline
│   │   └── model_loader.py        # Singleton loader
│   ├── api/
│   │   └── predict.py             # Prediction endpoint
│   ├── routers/
│   │   ├── auth.py                # Authentication endpoints
│   │   ├── observations.py        # Observation management
│   │   ├── species.py             # Species information
│   │   └── predict.py             # Prediction endpoints
│   ├── schemas/
│   │   └── prediction.py          # Pydantic models
│   └── services/
│       ├── pipeline_inference.py  # ML pipeline service
│       └── calibration.py         # Confidence calibration methods
├── environment.yml                # Conda environment spec
├── requirements.txt               # Pip dependencies (for Docker)
└── README.md
```

### Dependency Files

- **environment.yml**: Conda environment specification with all dependencies
- **requirements.txt**: Maintained for Docker builds and CI/CD pipelines

## Key Features

### Detection Deduplication
- **IoU-based merging**: Overlapping detections with IoU > 0.25 are merged
- **Center distance check**: Detections with centers within 50% of average box size are merged
- **Confidence-based selection**: Highest confidence detection is kept from each group

### Confidence Calibration
- **Temperature Scaling**: Divides logits by temperature (T=2.0) before softmax
- **Realistic Scores**: Prevents overly confident predictions (100% confidence)
- **External Calibrators**: Support for advanced calibration methods

### Supported Species
1. Bark Anole (*Anolis distichus*)
2. Brown Anole (*Anolis sagrei*)
3. Crested Anole (*Anolis cristatellus*)
4. Green Anole (*Anolis carolinensis*)
5. Knight Anole (*Anolis equestris*)

## Notes

- This backend supports both the full ML pipeline and frontend integration endpoints
- Mock endpoints are provided for development and testing
- Do not commit the `.venv` directory. A `.gitignore` is included to prevent this.
