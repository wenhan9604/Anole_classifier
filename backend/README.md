# Florida Anole Classifier - Backend API

FastAPI backend serving the 3-stage ML pipeline for anole species detection and classification.

## Architecture

### 3-Stage Pipeline

1. **Detection (YOLOv8)**: Detects lizards in uploaded images
2. **Cropping**: Extracts detected regions
3. **Classification (Swin Transformer)**: Classifies cropped images into 5 species

### Models

- **Detection**: Fine-tuned YOLOv8n
- **Classification**: Fine-tuned Swin Transformer (Base)

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
      "boundingBox": [120.5, 80.2, 250.3, 200.1],
      "detectionConfidence": 0.89
    }
  ],
  "processingTime": 1.23
}
```

- `GET /api/model-info` — Get information about loaded models
- `GET /api/health` — Health check endpoint

### Frontend Support
- `POST /api/observations` — Upload an observation (multipart: image + fields)
- `GET /api/observations` — List observations (mock)
- `GET /api/species/search?q=...` — Search species (mock)
- `GET /api/species/{scientific_name}` — Species details (mock)
- `POST /api/auth/mock-login` — Mock auth tokens

## Using the Spring_2025 Pipeline

The `/api/predict` endpoint can use the Spring_2025 pipeline (YOLOv8 detection + Swin classification) if the required models and packages are available. Configure via env vars and install optional deps:

1) Install optional ML dependencies

```bash
pip install ultralytics transformers torch Pillow
```

2) Provide model paths (defaults are shown):

```bash
export DETECTION_WEIGHTS_PATH=Spring_2025/runs/detect/train_yolov8n_v2/weights/best.pt
export CLASSIFICATION_MODEL_ID=swin-base-patch4-window12-384-finetuned-lizard-class-swin-base
```

3) Start the server. If models or deps are missing, the endpoint falls back to mock predictions.

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
│   ├── main.py              # FastAPI app
│   ├── models/
│   │   ├── pipeline.py      # 3-stage pipeline
│   │   └── model_loader.py  # Singleton loader
│   ├── api/
│   │   └── predict.py       # Prediction endpoint
│   ├── routers/
│   │   ├── auth.py          # Authentication endpoints
│   │   ├── observations.py  # Observation management
│   │   ├── species.py       # Species information
│   │   └── predict.py       # Prediction endpoints
│   ├── schemas/
│   │   └── prediction.py    # Pydantic models
│   └── services/
│       └── pipeline_inference.py  # ML pipeline service
├── environment.yml          # Conda environment spec
├── requirements.txt         # Pip dependencies (for Docker)
└── README.md
```

### Dependency Files

- **environment.yml**: Conda environment specification with all dependencies
- **requirements.txt**: Maintained for Docker builds and CI/CD pipelines

## Notes

- This backend supports both the full ML pipeline and frontend integration endpoints
- Mock endpoints are provided for development and testing
- Do not commit the `.venv` directory. A `.gitignore` is included to prevent this.
