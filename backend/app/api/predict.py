"""
Prediction API endpoint
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io
import logging
from typing import Dict, Any

from app.models.model_loader import model_loader
from app.schemas.prediction import PredictionResponse

# Increase PIL image size limit for high-resolution nature photos
Image.MAX_IMAGE_PIXELS = 500000000  # ~500 megapixels

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict anole species from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Detection and classification results
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Resize very large images for faster processing
        max_dimension = 4096
        if max(image.size) > max_dimension:
            logger.info(f"Resizing large image from {image.size}")
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}")
        
        # Get pipeline
        pipeline = model_loader.get_pipeline()
        
        # Run prediction
        result = pipeline.predict(image)
        
        logger.info(f"Prediction complete: {result['totalLizards']} lizard(s) detected")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/model-info")
async def model_info() -> Dict[str, Any]:
    """
    Get information about loaded models
    
    Returns:
        Model information and status
    """
    if not model_loader.is_loaded():
        return {
            "loaded": False,
            "message": "Models not yet loaded. Send a prediction request to load models."
        }
    
    pipeline = model_loader.get_pipeline()
    info = pipeline.get_model_info()
    info["loaded"] = True
    
    return info


