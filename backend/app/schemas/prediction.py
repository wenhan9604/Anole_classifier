"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionResult(BaseModel):
    """Individual prediction result for one detected lizard"""
    species: str = Field(..., description="Common name of the species")
    scientificName: str = Field(..., description="Scientific name of the species")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence (0-1)")
    count: int = Field(default=1, description="Count of this species in detection")
    boundingBox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    detectionConfidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detection confidence (0-1)")

class PredictionResponse(BaseModel):
    """Response from prediction endpoint"""
    totalLizards: int = Field(..., description="Total number of lizards detected")
    predictions: List[PredictionResult] = Field(..., description="List of predictions")
    processingTime: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "totalLizards": 2,
                "predictions": [
                    {
                        "species": "Green Anole",
                        "scientificName": "Anolis carolinensis",
                        "confidence": 0.94,
                        "count": 1,
                        "boundingBox": [120.5, 80.2, 250.3, 200.1],
                        "detectionConfidence": 0.89
                    },
                    {
                        "species": "Brown Anole",
                        "scientificName": "Anolis sagrei",
                        "confidence": 0.87,
                        "count": 1,
                        "boundingBox": [300.1, 150.5, 420.8, 280.3],
                        "detectionConfidence": 0.92
                    }
                ],
                "processingTime": 1.23
            }
        }


