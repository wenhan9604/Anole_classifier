# Docker Configuration

This directory contains all Docker-related configuration files for the Florida Anole Classifier project.

## Files

- **docker-compose.yml** - Orchestrates both backend and frontend services
- **backend.Dockerfile** - Container configuration for the FastAPI backend
- **frontend.Dockerfile** - Container configuration for the React frontend

## Quick Start

From the project root:

```bash
cd docker
docker-compose up --build
```

This will:
1. Build both backend and frontend containers
2. Start the services
3. Mount the required model directories
4. Expose the following ports:
   - Backend API: `http://localhost:8000`
   - Frontend: `http://localhost:5173`

## Services

### Backend Service
- **Base Image**: Python 3.11-slim
- **Port**: 8000
- **Health Check**: Enabled (30s interval)
- **Models**: Mounted from `../Spring_2025/` directory

### Frontend Service  
- **Base Image**: Node 20-alpine + nginx
- **Port**: 5173
- **Build**: Multi-stage (build + production)
- **Depends on**: Backend service

## Development Notes

The Docker setup is configured for production deployment. For local development, you may prefer using `start_dev.sh` for easier debugging and hot-reloading.

## Model Requirements

Ensure the following models exist before running:
- `../Spring_2025/runs/detect/train_yolov8n_v2/weights/best.pt` (YOLO detection)
- `../Spring_2025/swin-base-patch4-window12-384-finetuned-lizard-class-swin-base/` (Swin classification)

See `../backend/MODEL_SETUP.md` for instructions on obtaining the models.

