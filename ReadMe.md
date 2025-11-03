# Florida Anole Classifier

A full-stack web application for detecting and classifying Florida anole species from uploaded images using a 3-stage machine learning pipeline.

## What This Does

- **3-Stage ML Pipeline**: 
  1. YOLOv8x detection to locate lizards in images
  2. Automatic cropping of detected regions
  3. Swin Transformer classification for species identification
- **5 Species Support**: Green Anole, Brown Anole, Crested Anole, Knight Anole, Bark Anole
- **Multi-Detection**: Can detect and classify multiple lizards in a single image
- **Flexible Inference**: Choose between server-side (PyTorch/ONNX) or client-side (browser ONNX) inference
- **Confidence Scoring**: Shows confidence levels for each species prediction with visual indicators
- **Mobile Support**: Responsive design optimized for mobile devices

## Architecture

- **Backend**: FastAPI server with PyTorch and ONNX Runtime support
- **Frontend**: React + TypeScript with Vite
- **Models**: 
  - YOLOv8x for anole detection (640x640 input)
  - Swin Transformer Base for species classification (384x384 input)
- **Inference Modes**:
  - **CPU (Default)**: PyTorch on server CPU - balanced performance
  - **GPU**: PyTorch on server GPU - fastest for high-res images
  - **Client-side**: ONNX in browser (WebAssembly) - no server load

## Quick Start

### Prerequisites

**Backend:**
- Python 3.9+
- Conda (recommended)
- GPU with CUDA (optional, for GPU inference)

**Frontend:**
- Node.js 18+
- npm or yarn

### 1. Start the Backend

```bash
# Navigate to backend directory
cd backend

# Activate conda environment
conda activate anole-classifier

# Install dependencies (first time only)
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/api/docs`
- Health check: `http://localhost:8000/health`

### 2. Start the Frontend

```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start the development server
npm run dev
```

The application will be available at `http://localhost:5173`

## Inference Modes

The application supports three inference modes, accessible via URL parameters:

### 1. **CPU Mode (Default)** 
```
http://localhost:5173/predict
```
- Uses PyTorch on server CPU
- Good balance of speed and accuracy
- Recommended for most use cases
- No GPU required

### 2. **GPU Mode**
```
http://localhost:5173/predict?gpu=server
```
- Uses PyTorch on server GPU (if available)
- Fastest inference for high-resolution images
- Requires CUDA-compatible GPU
- Best for batch processing

### 3. **Client-Side Mode**
```
http://localhost:5173/predict?gpu=client-side
```
- Runs ONNX models directly in your browser (WebAssembly)
- No server load, privacy-preserving
- Works offline after initial model download (~620 MB)
- Slower but doesn't require server GPU

## Project Structure

```
Anole_classifier/
├── backend/                       # FastAPI backend
│   ├── app/
│   │   ├── models/               # ML model loading and pipeline
│   │   │   ├── pipeline.py       # 3-stage pipeline orchestration
│   │   │   └── model_loader.py   # Singleton model loader
│   │   ├── routers/              # API endpoints
│   │   │   └── predict.py        # Prediction endpoints
│   │   ├── services/             # Business logic
│   │   │   ├── pipeline_inference.py      # PyTorch inference
│   │   │   └── onnx_pipeline_inference.py # ONNX inference
│   │   └── main.py               # FastAPI app initialization
│   └── requirements.txt          # Python dependencies
│
├── frontend/                      # React frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── LandingPage.tsx   # Welcome screen
│   │   │   └── PredictionPage.tsx # Main classification interface
│   │   ├── services/
│   │   │   ├── OnnxDetectionService.ts   # Browser ONNX inference
│   │   │   ├── AnoleDetectionService.ts  # Unified detection API
│   │   │   └── config.ts                 # API configuration
│   │   └── App.tsx               # Main app component
│   ├── public/                   # ONNX WASM files
│   └── package.json              # Node dependencies
│
├── Spring_2025/
│   ├── models/                   # Trained ML models
│   │   ├── yolov8x/             # YOLOv8x detection models
│   │   │   ├── best.pt          # PyTorch weights
│   │   │   └── best.onnx        # ONNX format
│   │   ├── swin_transformer_base_lizard_v4/  # Swin classification
│   │   ├── yolo_best.onnx       # Standalone YOLO ONNX
│   │   └── swin_model.onnx      # Standalone Swin ONNX
│   └── *.ipynb                  # Training notebooks
│
└── docs/                         # Documentation
    ├── ONNX_SETUP.md            # ONNX setup guide
    ├── API_USAGE_GUIDE.md       # API usage examples
    └── ...
```

## API Usage Examples

### Python
```python
import requests

# Default CPU inference
with open("lizard.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/predict",
        files={"file": f}
    )
print(response.json())

# GPU inference
response = requests.post(
    "http://localhost:8000/api/predict?gpu=server",
    files={"file": ("lizard.jpg", open("lizard.jpg", "rb"))}
)
```

### cURL
```bash
# Default CPU inference
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@lizard.jpg"

# GPU inference
curl -X POST "http://localhost:8000/api/predict?gpu=server" \
  -F "file=@lizard.jpg"

# ONNX inference
curl -X POST "http://localhost:8000/api/predict?use_onnx=true" \
  -F "file=@lizard.jpg"
```

## Model Information

- **YOLOv8x Detection**: Single-class object detection model trained on anole images
- **Swin Transformer Base**: Multi-class classification model for 5 anole species
- **Model Size**: ~750 MB total (620 MB for ONNX models)
- **Training Data**: Custom dataset of Florida anole species

For more details, see [`Spring_2025/models/README.md`](Spring_2025/models/README.md)

## Documentation

- [ONNX Setup Guide](docs/ONNX_SETUP.md) - Complete guide for ONNX inference
- [API Usage Guide](docs/API_USAGE_GUIDE.md) - Detailed API documentation
- [Frontend Quick Start](docs/FRONTEND_QUICKSTART.md) - Frontend development guide

## License

[Add license information here]
