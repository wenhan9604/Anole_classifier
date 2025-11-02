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

