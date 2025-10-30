from typing import List, Optional
import logging

from fastapi import APIRouter, File, UploadFile, HTTPException
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


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    data = await file.read()
    # Use pipeline-based inference (mock fallback disabled)
    try:
        from ..services.pipeline_inference import predict_image_bytes

        result = predict_image_bytes(data)
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
        return DetectionResult(totalLizards=int(result.get("totalLizards", len(preds))), predictions=preds)
    except Exception as e:
        # Mock fallback disabled - raise actual error
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}. Please ensure ML dependencies (torch, transformers, ultralytics) are installed and models are available."
        )
