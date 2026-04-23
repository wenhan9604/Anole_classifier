import os
import time
import httpx
import logging
import asyncio
from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv

# Try to load from backend/.env and ../.env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

router = APIRouter()
logger = logging.getLogger(__name__)

# Cache TTL in seconds (5 minutes)
CACHE_TTL = 300
_stats_cache = {
    "data": {
        "observations": 0,
        "species": 0,
        "contributors": 0
    },
    "last_updated": 0
}

class AppStatsResponse(BaseModel):
    observations: int
    species: int
    contributors: int

async def fetch_stat(client, endpoint, params):
    try:
        response = await client.get(
            f"https://api.inaturalist.org/v1/observations{endpoint}",
            params=params,
            timeout=10.0
        )
        if response.status_code == 200:
            return response.json().get("total_results", 0)
        else:
            logger.error(f"iNaturalist API {endpoint} returned {response.status_code}: {response.text}")
            return 0
    except Exception as e:
        logger.error(f"Error fetching {endpoint} from iNaturalist: {e}")
        return 0

@router.get("/metrics/stats", response_model=AppStatsResponse)
async def get_app_stats():
    global _stats_cache
    current_time = time.time()
    
    # Return cached value if within TTL
    if current_time - _stats_cache["last_updated"] < CACHE_TTL:
        return AppStatsResponse(**_stats_cache["data"])
        
    app_id = os.getenv("INAT_NUMERIC_APP_ID")
    if not app_id:
        logger.warning("INAT_NUMERIC_APP_ID not found in environment.")
        return AppStatsResponse(**_stats_cache["data"])
        
    try:
        async with httpx.AsyncClient() as client:
            # Fetch all three metrics in parallel
            obs_task = fetch_stat(client, "", {"oauth_application_id": app_id, "per_page": 0})
            species_task = fetch_stat(client, "/species_counts", {"oauth_application_id": app_id, "per_page": 0})
            observers_task = fetch_stat(client, "/observers", {"oauth_application_id": app_id, "per_page": 0})
            
            obs_count, species_count, observers_count = await asyncio.gather(
                obs_task, species_task, observers_task
            )
            
            new_data = {
                "observations": obs_count,
                "species": species_count,
                "contributors": observers_count
            }
            
            # Update cache
            _stats_cache["data"] = new_data
            _stats_cache["last_updated"] = current_time
            
            return AppStatsResponse(**new_data)
                
    except Exception as e:
        logger.error(f"Error in get_app_stats: {e}")
        return AppStatsResponse(**_stats_cache["data"])

# Keep the old endpoint for backward compatibility (optional, but good practice)
@router.get("/metrics/observations_count")
async def get_observations_count_compat():
    stats = await get_app_stats()
    return {"count": stats.observations}
