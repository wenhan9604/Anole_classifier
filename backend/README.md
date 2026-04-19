# Florida Anole Classifier - Backend API

FastAPI backend serving the 3-stage ML pipeline for anole species detection and classification.

## Architecture

### 3-Stage Pipeline

1. **Detection (YOLOv8)**: Detects lizards in uploaded images
2. **Cropping**: Extracts detected regions
3. **Classification (Swin Transformer)**: Classifies cropped images into 5 species

### Models

- **Detection**: Fine-tuned YOLOv8n :`Spring_2025/models/train_yolov8n_v2/weights/best.pt`
- **Classification**: Fine-tuned Swin Transformer: `Spring_2025/models/swin_transformer_base_lizard_v4`

## Setup

### Model Files

Model weights need to be downloaded:

1. **Download model files** 

2. **Place them in the following locations:**
   - YOLO: `Spring_2025/models/train_yolov8n_v2/weights/best.pt`
   - Swin: `Spring_2025/models/swin_transformer_base_lizard_v4/` (all files including `model.safetensors`, `config.json`, etc.)

The backend will automatically detect and use these models when you start the server.

## Quick Start

### Launch (backend-integration branch)

Use these exact commands to run the app locally for this branch.

Backend (FastAPI):

```bash
cd /Users/anqizheng/Desktop/Anole_classifier/backend
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install "uvicorn[standard]" fastapi Pillow
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend (Vite/React):

```bash
cd /Users/anqizheng/Desktop/Anole_classifier/frontend
npm install
npm run dev -- --host --port 5173
```

Open the UI at `http://localhost:5173`. The frontend calls the backend at `http://localhost:8000`.

## iNaturalist OAuth

Register an app at [iNaturalist OAuth applications](https://www.inaturalist.org/oauth/applications). Set environment variables (see `.env.example`):

| Variable | Purpose |
|----------|---------|
| `INAT_CLIENT_ID` | OAuth client id |
| `INAT_CLIENT_SECRET` | OAuth secret (server only) |
| `INAT_REDIRECT_URI` | Must match the registered callback URL, e.g. `https://api.yoursite.com/api/auth/inat/callback` |
| `INAT_FRONTEND_SUCCESS_URL` | Where to send the browser after success, e.g. `https://app.yoursite.com/predict?inat=connected` |
| `INAT_SCOPES` | Optional space-separated scopes |
| `INAT_COOKIE_SAMESITE` | `lax` (same-site) or `none` (cross-site API + frontend; requires HTTPS) |

Endpoints:

- `GET /api/auth/inat/login` — starts OAuth (sets HTTP-only session cookie)
- `GET /api/auth/inat/callback` — iNaturalist redirects here with `code` and `state`
- `GET /api/auth/inat/status` — `{ "connected": bool, "expiresAt": number | null }`
- `POST /api/auth/inat/logout` — clears server tokens and cookie

`POST /api/observations` requires a connected session (same cookie). Ensure `CORS_ORIGINS` includes your frontend origin and the frontend uses `credentials: 'include'` (`VITE_API_BASE_URL`).

Dev-only: `ENABLE_INAT_MOCK_AUTH=true` enables `POST /api/auth/mock-login` (disabled by default).

Run OAuth tests: `cd backend && pytest tests/test_inat_oauth.py -v`

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

