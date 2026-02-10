"""
CCTV Analytics App Package
==========================
FastAPI app factory with lifespan, CORS, and router registration.
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.database import database
from app.video.models import preload_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    # Pre-load models in background thread to not block startup
    preload_thread = threading.Thread(target=preload_models, daemon=True)
    preload_thread.start()
    yield
    await database.disconnect()


app = FastAPI(
    title="Video Analytics MVP",
    description="Privacy-preserving venue analytics API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
from app.routers import health, venues, events, analytics, advanced_analytics  # noqa: E402
from app.routers import behavior, alerts, benchmarks, visitors  # noqa: E402
from app.routers import video_processing, batch, map_api  # noqa: E402

app.include_router(health.router)
app.include_router(venues.router)
app.include_router(events.router)
app.include_router(analytics.router)
app.include_router(advanced_analytics.router)
app.include_router(behavior.router)
app.include_router(alerts.router)
app.include_router(benchmarks.router)
app.include_router(visitors.router)
app.include_router(video_processing.router)
app.include_router(batch.router)
app.include_router(map_api.router)

# Pages router (HTML endpoints) - imported last since it may reference other routers
try:
    from app.routers import pages  # noqa: E402
    app.include_router(pages.router)
except ImportError:
    pass  # pages.py may not exist yet during development
