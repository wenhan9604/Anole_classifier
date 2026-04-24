import os
import time
import httpx
import logging
import asyncio
from typing import List, Optional, Dict
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Try to load from backend/.env and ../.env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

router = APIRouter()
logger = logging.getLogger(__name__)

# Cache TTL in seconds (10 minutes for detailed dashboard)
DASHBOARD_CACHE_TTL = 600
_dashboard_cache = {
    "data": None,
    "last_updated": 0
}

class AppStatsResponse(BaseModel):
    observations: int
    species: int
    contributors: int

class SpeciesStat(BaseModel):
    name: str
    count: int
    id: int

class ObserverStat(BaseModel):
    login: str
    count: int
    icon_url: Optional[str] = None

class DashboardStatsResponse(BaseModel):
    observations: int
    species: int
    contributors: int
    activity: List[int]
    top_observers: List[ObserverStat]
    species_distribution: List[SpeciesStat]

async def fetch_total(client, endpoint, params):
    try:
        response = await client.get(
            f"https://api.inaturalist.org/v1/observations{endpoint}",
            params=params,
            timeout=10.0
        )
        if response.status_code == 200:
            return response.json().get("total_results", 0)
        return 0
    except Exception as e:
        logger.error(f"Error fetching total from {endpoint}: {e}")
        return 0

async def fetch_results(client, endpoint, params):
    try:
        response = await client.get(
            f"https://api.inaturalist.org/v1/observations{endpoint}",
            params=params,
            timeout=10.0
        )
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except Exception as e:
        logger.error(f"Error fetching results from {endpoint}: {e}")
        return []

@router.get("/metrics/stats", response_model=AppStatsResponse)
async def get_app_stats():
    # Simple version for backward compat or quick checks
    app_id = os.getenv("INAT_NUMERIC_APP_ID")
    if not app_id: return AppStatsResponse(observations=0, species=0, contributors=0)
    
    async with httpx.AsyncClient() as client:
        params = {"oauth_application_id": app_id, "per_page": 0}
        obs_task = fetch_total(client, "", params)
        species_task = fetch_total(client, "/species_counts", params)
        observers_task = fetch_total(client, "/observers", params)
        
        obs_count, species_count, observers_count = await asyncio.gather(obs_task, species_task, observers_task)
        return AppStatsResponse(observations=obs_count, species=species_count, contributors=observers_count)

@router.get("/metrics/dashboard", response_model=DashboardStatsResponse)
async def get_dashboard_stats():
    global _dashboard_cache
    current_time = time.time()
    
    if _dashboard_cache["data"] and current_time - _dashboard_cache["last_updated"] < DASHBOARD_CACHE_TTL:
        return _dashboard_cache["data"]
        
    app_id = os.getenv("INAT_NUMERIC_APP_ID")
    if not app_id:
        return DashboardStatsResponse(observations=0, species=0, contributors=0, activity=[], top_observers=[], species_distribution=[])
        
    try:
        async with httpx.AsyncClient() as client:
            params = {"oauth_application_id": app_id}
            
            # 1. Totals
            obs_task = fetch_total(client, "", {**params, "per_page": 0})
            
            # 2. Species Distribution
            species_task = fetch_results(client, "/species_counts", {**params, "per_page": 10})
            
            # 3. Top Observers
            observers_task = fetch_results(client, "/observers", {**params, "per_page": 5})
            
            # 4. Activity Histogram
            # We want last 30 days
            histogram_task = client.get(
                "https://api.inaturalist.org/v1/observations/histogram",
                params={**params, "date_field": "observed", "interval": "day"},
                timeout=10.0
            )
            
            obs_total, species_results, observer_results, histogram_res = await asyncio.gather(
                obs_task, species_task, observers_task, histogram_task
            )
            
            # Process Activity
            activity = []
            if histogram_res.status_code == 200:
                h_data = histogram_res.json().get("results", {}).get("day", {})
                # Fill last 30 days
                today = datetime.now()
                for i in range(29, -1, -1):
                    date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                    activity.append(h_data.get(date_str, 0))
            
            # Process Species
            species_distribution = [
                SpeciesStat(
                    name=s.get("taxon", {}).get("preferred_common_name") or s.get("taxon", {}).get("name") or "Unknown",
                    count=s.get("count", 0),
                    id=s.get("taxon", {}).get("id", 0)
                ) for s in species_results
            ]
            
            # Process Observers
            top_observers = [
                ObserverStat(
                    login=o.get("user", {}).get("login", "Unknown"),
                    count=o.get("observation_count", 0),
                    icon_url=o.get("user", {}).get("icon_url")
                ) for o in observer_results
            ]
            
            data = DashboardStatsResponse(
                observations=obs_total,
                species=len(species_results),
                contributors=len(observer_results),
                activity=activity,
                top_observers=top_observers,
                species_distribution=species_distribution
            )
            
            _dashboard_cache["data"] = data
            _dashboard_cache["last_updated"] = current_time
            return data
            
    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {e}")
        return DashboardStatsResponse(observations=0, species=0, contributors=0, activity=[], top_observers=[], species_distribution=[])
