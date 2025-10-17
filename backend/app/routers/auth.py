from fastapi import APIRouter
from pydantic import BaseModel
from time import time


router = APIRouter()


class AuthResponse(BaseModel):
    accessToken: str
    refreshToken: str
    expiresAt: int


@router.post("/auth/mock-login", response_model=AuthResponse)
async def mock_login():
    now = int(time())
    # Mock tokens for local development
    return AuthResponse(
        accessToken="mock_access_token",
        refreshToken="mock_refresh_token",
        expiresAt=now + 3600,
    )

