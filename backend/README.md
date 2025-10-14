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

## Installation

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

### Endpoints

#### POST `/api/predict`
Upload an image for prediction

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

#### GET `/api/model-info`
Get information about loaded models

#### GET `/api/health`
Health check endpoint

## Testing

```bash
# Test with curl
curl -X POST "http://localhost:8000/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/anole_image.jpg"
```

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
│   └── schemas/
│       └── prediction.py    # Pydantic models
├── environment.yml          # Conda environment spec
├── requirements.txt         # Pip dependencies (for Docker)
└── README.md
```

### Dependency Files

- **environment.yml**: Conda environment specification with all dependencies
- **requirements.txt**: Maintained for Docker builds and CI/CD pipelines
