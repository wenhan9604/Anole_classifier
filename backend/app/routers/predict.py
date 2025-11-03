from typing import List, Optional
import logging

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict


router = APIRouter()
logger = logging.getLogger(__name__)


class Prediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    species: str
    scientific_name: str = Field(alias="scientificName")
    confidence: float
    count: int
    box: Optional[List[float]] = None  # [x1, y1, x2, y2] bounding box coordinates


class DetectionResult(BaseModel):
    totalLizards: int
    predictions: List[Prediction]
    modelType: str = "pytorch"  # "pytorch" or "onnx"
    processingTime: Optional[float] = None


@router.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    try:
        from ..services.onnx_pipeline_inference import get_onnx_info, is_onnx_available
        
        onnx_info = get_onnx_info()
        
        return {
            "pytorch": {
                "available": True,
                "type": "PyTorch"
            },
            "onnx": onnx_info
        }
    except Exception as e:
        return {
            "pytorch": {"available": True},
            "onnx": {"available": False, "error": str(e)}
        }


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_onnx: bool = Query(False, description="Use ONNX models instead of PyTorch (default: False, uses PyTorch CPU)")
):
    """
    Predict anole species in an image (Backend inference)
    
    Args:
        file: Image file (jpg, png, etc.)
        use_onnx: If True, use ONNX models; if False, use PyTorch (default: False)
                  PyTorch runs on CPU by default, GPU if available
    
    Returns:
        Detection results with species classification and bounding boxes
    
    Note: Default backend uses PyTorch CPU (best.pt models).
          Add ?use_onnx=true for ONNX CPU inference.
          Use frontend ?gpu=client-side for browser-based inference.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    data = await file.read()
    
    try:
        if use_onnx:
            # Use ONNX-enabled pipeline
            from ..services.onnx_pipeline_inference import predict_image_bytes_onnx
            
            result = predict_image_bytes_onnx(data)
            model_type = "onnx-cpu"
        else:
            # Use PyTorch pipeline (default)
            from ..services.pipeline_inference import predict_image_bytes
            
            result = predict_image_bytes(data)
            model_type = "pytorch-cpu"
        
        preds: list[Prediction] = [
            Prediction(
                species=p["species"],
                scientific_name=p["scientificName"],
                confidence=float(p["confidence"]),
                count=int(p.get("count", 1)),
                box=p.get("box"),  # Include bounding box coordinates
            )
            for p in result.get("predictions", [])
        ]
        
        return DetectionResult(
            totalLizards=int(result.get("totalLizards", len(preds))),
            predictions=preds,
            modelType=model_type,
            processingTime=result.get("processingTime")
        )
        
    except Exception as e:
        # Raise actual error with helpful message
        logger.error(f"Prediction failed (use_onnx={use_onnx}): {str(e)}", exc_info=True)
        
        error_msg = f"Prediction failed: {str(e)}."
        if use_onnx and "onnxruntime" in str(e).lower():
            error_msg += " ONNX Runtime not installed. Install with: pip install onnxruntime"
        elif use_onnx and "not found" in str(e).lower():
            error_msg += " ONNX models not found. Ensure yolo_best.onnx and swin_model.onnx are in backend/models/"
        else:
            error_msg += " Please ensure ML dependencies are installed."
        
        raise HTTPException(status_code=500, detail=error_msg)
