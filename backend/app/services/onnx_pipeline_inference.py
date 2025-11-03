"""
ONNX-based inference using the pipeline.py with ONNX models
Provides faster CPU inference compared to PyTorch
"""

import os
import time
from typing import Any, Dict
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# Cache the pipeline instance
_pipeline = None


def _get_onnx_model_paths():
    """Get paths to ONNX models (relative to backend directory)"""
    # Check environment variables first
    # Navigate from backend/app/services/ -> backend/models/
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    yolo_path = os.getenv("ONNX_YOLO_PATH", os.path.join(base_dir, "models", "yolo_best.onnx"))
    swin_path = os.getenv("ONNX_SWIN_PATH", os.path.join(base_dir, "models", "swin_model.onnx"))
    
    # Verify files exist
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"ONNX YOLO model not found at {yolo_path}")
    if not os.path.exists(swin_path):
        raise FileNotFoundError(f"ONNX Swin model not found at {swin_path}")
    
    return yolo_path, swin_path


def _load_onnx_pipeline():
    """Load the ONNX pipeline (lazy loading, cached)"""
    global _pipeline
    
    if _pipeline is not None:
        return _pipeline
    
    try:
        from ..models.pipeline import AnolePipeline
    except ImportError:
        from app.models.pipeline import AnolePipeline
    
    yolo_path, swin_path = _get_onnx_model_paths()
    
    logger.info(f"Loading ONNX pipeline...")
    logger.info(f"  YOLO: {yolo_path}")
    logger.info(f"  Swin: {swin_path}")
    
    _pipeline = AnolePipeline(
        yolo_model_path=yolo_path,
        swin_model_path=swin_path,
        conf_threshold=0.25,
        iou_threshold=0.5,
        use_onnx=True  # Enable ONNX mode
    )
    
    logger.info("ONNX pipeline loaded successfully")
    logger.info(f"Model info: {_pipeline.get_model_info()}")
    
    return _pipeline


def predict_image_bytes_onnx(image_bytes: bytes) -> Dict[str, Any]:
    """
    Run ONNX-based detection + classification on an image
    
    Args:
        image_bytes: Image data as bytes
    
    Returns:
        Dictionary with detection results:
        {
            "totalLizards": int,
            "predictions": [
                {
                    "species": str,
                    "scientificName": str,
                    "confidence": float,
                    "count": int,
                    "box": [x1, y1, x2, y2],
                    "detectionConfidence": float
                }
            ],
            "processingTime": float
        }
    """
    start_time = time.time()
    
    # Load pipeline (cached after first call)
    pipeline = _load_onnx_pipeline()
    
    # Load image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    logger.info(f"Processing image: {image.size}")
    
    # Run prediction
    results = pipeline.predict(image)
    
    processing_time = time.time() - start_time
    results["processingTime"] = processing_time
    
    logger.info(f"ONNX prediction complete: {results['totalLizards']} lizards detected in {processing_time:.2f}s")
    
    return results


def is_onnx_available() -> bool:
    """Check if ONNX runtime is available"""
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


def get_onnx_info() -> Dict[str, Any]:
    """Get information about ONNX setup"""
    try:
        yolo_path, swin_path = _get_onnx_model_paths()
        onnx_available = is_onnx_available()
        
        return {
            "available": onnx_available,
            "yolo_path": yolo_path,
            "yolo_exists": os.path.exists(yolo_path),
            "swin_path": swin_path,
            "swin_exists": os.path.exists(swin_path),
            "pipeline_loaded": _pipeline is not None
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

