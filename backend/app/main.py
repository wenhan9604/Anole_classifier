"""
FastAPI backend for Florida Anole Species Classification
Serves the 3-stage ML pipeline: Detection -> Cropping -> Classification
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path

from app.api.predict import router as predict_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Florida Anole Classifier API",
    description="3-stage ML pipeline for detecting and classifying Florida anole species",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for frontend (allow all localhost ports for development)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",  # Allow any localhost port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, prefix="/api", tags=["prediction"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Florida Anole Classifier API is running",
        "version": "1.0.0"
    }

@app.get("/api/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": False,  # Will be updated by model loader
        "gpu_available": False   # Will be updated by model loader
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

