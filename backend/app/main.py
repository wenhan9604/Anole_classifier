"""
FastAPI backend for Florida Anole Species Classification
Serves the 3-stage ML pipeline: Detection -> Cropping -> Classification
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import observations, species, auth, predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,https://lizard-class.serveo.net,https://*.ngrok.io,https://*.ngrok-free.app,https://*.ngrok-free.dev")
    return [o.strip() for o in raw.split(",") if o.strip()]


# Initialize FastAPI app
app = FastAPI(
    title="Florida Anole Classifier API",
    description="3-stage ML pipeline for detecting and classifying Florida anole species",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Florida Anole Classifier API is running",
        "version": "1.0.0"
    }

@app.get("/health")
def health() -> dict:
    """Basic health check endpoint"""
    return {"status": "ok"}

@app.get("/api/health")
async def api_health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": False,  # Will be updated by model loader
        "gpu_available": False   # Will be updated by model loader
    }

# Include routers
app.include_router(observations.router, prefix="/api", tags=["observations"])
app.include_router(species.router, prefix="/api", tags=["species"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(predict.router, prefix="/api", tags=["predict"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

