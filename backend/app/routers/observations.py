from datetime import date
from typing import Optional
from uuid import uuid4
import logging

import httpx

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, ConfigDict

from ..services.inat_request_auth import resolve_inat_authorization_header

logger = logging.getLogger(__name__)

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
    request: Request,
    file: UploadFile = File(...),
    species: str = Form(...),
    scientific_name: str = Form(..., alias="scientificName"),
    confidence: float = Form(...),
    count: int = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    notes: Optional[str] = Form(None),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    file_content = await file.read()

    auth_header = await resolve_inat_authorization_header(request)
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Not connected to iNaturalist. Open Connect iNaturalist in the app to authorize.",
        )

    obs_data = {
        "observation": {
            "species_guess": scientific_name,
            "description": f"{notes or ''}\n\nIdentified as {species} with {confidence:.1%} confidence by Lizard Lens.".strip(),
            "observed_on_string": date.today().isoformat(),
        }
    }

    if latitude is not None and longitude is not None:
        obs_data["observation"]["latitude"] = latitude
        obs_data["observation"]["longitude"] = longitude

    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.inaturalist.org/v1/observations",
                json=obs_data,
                headers=headers,
                timeout=60.0,
            )

            if response.status_code not in (200, 201):
                error_detail = response.text
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"iNaturalist API error: {error_detail}",
                )

            irat_response = response.json()

            if "id" in irat_response:
                inat_id = irat_response["id"]
            elif "results" in irat_response and len(irat_response["results"]) > 0:
                inat_id = irat_response["results"][0]["id"]
            else:
                inat_id = str(uuid4())
                logger.warning("Unexpected iNat observation create response shape: %s", irat_response)

            files = {"file": (file.filename, file_content, file.content_type)}
            photo_data = {"observation_photo[observation_id]": inat_id}
            photo_headers = {"Authorization": auth_header}

            photo_response = await client.post(
                "https://api.inaturalist.org/v1/observation_photos",
                data=photo_data,
                files=files,
                headers=photo_headers,
                timeout=120.0,
            )

            if photo_response.status_code not in (200, 201):
                logger.error("Failed to upload photo to iNat: %s", photo_response.text)

            return ObservationResponse(
                success=True,
                id=str(inat_id),
                species=species,
                scientific_name=scientific_name,
                confidence=confidence,
                count=count,
                latitude=latitude,
                longitude=longitude,
                notes=notes,
            )

        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to communicate with iNaturalist: {str(exc)}",
            ) from exc


class ObservationItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    species: str
    scientific_name: str = Field(alias="scientificName")
    confidence: float
    count: int


@router.get("/observations", response_model=list[ObservationItem])
async def list_observations():
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
