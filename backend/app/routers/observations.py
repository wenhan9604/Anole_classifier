from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel, Field, ConfigDict


router = APIRouter()


class ObservationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    success: bool
    id: str
    species: str
    scientific_name: str = Field(alias="scientificName")
    confidence: float
    count: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: Optional[str] = None


@router.post("/observations", response_model=ObservationResponse)
async def create_observation(
    file: UploadFile = File(...),
    species: str = Form(...),
    scientific_name: str = Form(..., alias="scientificName"),
    confidence: float = Form(...),
    count: int = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    notes: Optional[str] = Form(None),
):
    # Basic validation for file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    # This is a mock endpoint. In a real app you might store the file
    # and forward the observation to iNaturalist or your DB.
    _ = await file.read()  # read to exercise the upload path; discard

    return ObservationResponse(
        success=True,
        id=str(uuid4()),
        species=species,
        scientific_name=scientific_name,
        confidence=confidence,
        count=count,
        latitude=latitude,
        longitude=longitude,
        notes=notes,
    )


class ObservationItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    species: str
    scientific_name: str = Field(alias="scientificName")
    confidence: float
    count: int


@router.get("/observations", response_model=list[ObservationItem])
async def list_observations():
    # Mock data
    return [
        ObservationItem(
            id="obs_1",
            species="Green Anole",
            scientific_name="Anolis carolinensis",
            confidence=0.92,
            count=1,
        ),
        ObservationItem(
            id="obs_2",
            species="Brown Anole",
            scientific_name="Anolis sagrei",
            confidence=0.81,
            count=2,
        ),
    ]
