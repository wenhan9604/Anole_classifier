import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import observations, species, auth, predict


def get_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="Anole Classifier API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Routers
app.include_router(observations.router, prefix="/api", tags=["observations"])
app.include_router(species.router, prefix="/api", tags=["species"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(predict.router, prefix="/api", tags=["predict"])

