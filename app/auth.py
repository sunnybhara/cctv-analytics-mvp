"""
API Key Authentication
======================
FastAPI dependency that validates X-API-Key header against venues table.
"""

import threading
from typing import Optional

import sqlalchemy
from fastapi import Header, HTTPException

from cachetools import TTLCache

from app.config import AUTH_ENABLED
from app.database import database, venues

_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
_lock = threading.Lock()


def _check_cache(api_key: str) -> Optional[str]:
    with _lock:
        return _cache.get(api_key)


def _set_cache(api_key: str, venue_id: str):
    with _lock:
        _cache[api_key] = venue_id


async def require_api_key(x_api_key: str = Header(default=None)) -> str:
    """
    Validate API key and return the associated venue_id.
    Raises 401 if key is missing or invalid.
    When AUTH_ENABLED=False, returns a dummy venue_id for dev/testing.
    """
    if not AUTH_ENABLED:
        return "__dev__"

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Check cache first
    cached = _check_cache(x_api_key)
    if cached is not None:
        return cached

    # Cache miss — query DB
    query = sqlalchemy.select(venues.c.id).where(venues.c.api_key == x_api_key)
    row = await database.fetch_one(query)

    if not row:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    venue_id = row["id"]
    _set_cache(x_api_key, venue_id)
    return venue_id
