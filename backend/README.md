# Anole Classifier Backend (FastAPI)

A lightweight FastAPI backend to support the frontend with endpoints for observations, species info, prediction (mock), and auth (mock). Includes CORS for local dev and file upload handling.

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

## Endpoints

- `GET /health` — basic health check
- `POST /api/observations` — upload an observation (multipart: image + fields)
- `GET /api/observations` — list observations (mock)
- `GET /api/species/search?q=...` — search species (mock)
- `GET /api/species/{scientific_name}` — species details (mock)
- `POST /api/predict` — mock classifier returning detection results
- `POST /api/auth/mock-login` — mock auth tokens

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

## CORS

By default, CORS allows `http://localhost:5173`. To override, set `CORS_ORIGINS` env var with comma-separated origins.

## Notes

- This is a scaffold with mocked data. Replace mocks with real integrations as needed.
- Do not commit the `.venv` directory. A `.gitignore` is included to prevent this.
