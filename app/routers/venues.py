"""
Venue CRUD Endpoints
====================
Create and list venues with optional geo-location, zone calibration, and embedding management.
"""

import json
import secrets
from datetime import datetime

import sqlalchemy
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from app.auth import require_api_key, verify_venue_access

from app.database import database, venues, events, alerts, jobs, visitor_embeddings
from app.responses import success_response
from app.schemas import VenueCreate
from app.video.helpers import lat_long_to_h3

MAX_ZONES_PER_VENUE = 10


class ZonePolygon(BaseModel):
    name: str
    color: str = "#3b82f6"
    points: List[List[float]]


class ZoneConfig(BaseModel):
    zones: List[ZonePolygon]
    reference_frame_width: int = 640
    reference_frame_height: int = 480

router = APIRouter()


@router.post("/venues")
async def create_venue(venue: VenueCreate):
    """Register a new venue with optional geo-location."""
    api_key = secrets.token_hex(32)

    # Calculate H3 zone if lat/long provided
    h3_zone = None
    if venue.latitude is not None and venue.longitude is not None:
        h3_zone = lat_long_to_h3(venue.latitude, venue.longitude)

    query = venues.insert().values(
        id=venue.id,
        name=venue.name,
        api_key=api_key,
        created_at=datetime.utcnow(),
        latitude=venue.latitude,
        longitude=venue.longitude,
        h3_zone=h3_zone,
        address=venue.address,
        city=venue.city,
        country=venue.country,
        venue_type=venue.venue_type
    )

    try:
        await database.execute(query)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Venue with this ID already exists")

    return success_response({
        "venue_id": venue.id,
        "api_key": api_key,
        "h3_zone": h3_zone,
        "message": "Save this API key - it won't be shown again"
    })


@router.get("/venues")
async def list_venues(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0)
):
    """List all venues with geo info (without API keys)."""
    from sqlalchemy import func

    # Get total count
    total = await database.fetch_val(
        sqlalchemy.select(func.count()).select_from(venues)
    ) or 0

    query = sqlalchemy.select(
        venues.c.id, venues.c.name, venues.c.created_at,
        venues.c.latitude, venues.c.longitude, venues.c.h3_zone,
        venues.c.city, venues.c.country, venues.c.venue_type
    ).limit(limit).offset(offset)
    rows = await database.fetch_all(query)
    items = [{
        "id": r["id"],
        "name": r["name"],
        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        "latitude": r["latitude"],
        "longitude": r["longitude"],
        "h3_zone": r["h3_zone"],
        "city": r["city"],
        "country": r["country"],
        "venue_type": r["venue_type"]
    } for r in rows]
    return success_response(items, pagination={
        "limit": limit, "offset": offset, "total": total,
        "has_more": offset + limit < total
    })


@router.delete("/venues/{venue_id}")
async def delete_venue(venue_id: str, auth_venue_id: str = Depends(require_api_key)):
    """Delete a venue and all its associated data (including orphaned records)."""
    verify_venue_access(auth_venue_id, venue_id)
    # Always clean associated data even if venue row is missing
    await database.execute(events.delete().where(events.c.venue_id == venue_id))
    await database.execute(alerts.delete().where(alerts.c.venue_id == venue_id))
    await database.execute(jobs.delete().where(jobs.c.venue_id == venue_id))
    await database.execute(visitor_embeddings.delete().where(visitor_embeddings.c.venue_id == venue_id))
    await database.execute(venues.delete().where(venues.c.id == venue_id))

    return success_response({"message": f"Venue '{venue_id}' and all associated data deleted"})


@router.post("/venues/{venue_id}/purge-embeddings")
async def purge_venue_embeddings(
    venue_id: str,
    older_than_days: int = Query(default=90, ge=1),
    _api_key: str = Depends(require_api_key),
):
    """Purge expired visitor embeddings for a venue (GDPR right to erasure)."""
    from app.video.embeddings import purge_expired_embeddings_sync
    count = purge_expired_embeddings_sync(venue_id=venue_id, retention_days=older_than_days)
    return success_response({"purged": count, "older_than_days": older_than_days})


@router.get("/venues/{venue_id}/zones")
async def get_venue_zones(venue_id: str, _api_key: str = Depends(require_api_key)):
    """Get zone calibration config for a venue."""
    row = await database.fetch_one(
        sqlalchemy.select(venues.c.config).where(venues.c.id == venue_id)
    )
    if not row:
        raise HTTPException(status_code=404, detail="Venue not found")
    config = row["config"] or {}
    if isinstance(config, str):
        config = json.loads(config)
    return success_response(config.get("zones_config", {"zones": []}))


@router.put("/venues/{venue_id}/zones")
async def save_venue_zones(
    venue_id: str,
    zone_config: ZoneConfig,
    _api_key: str = Depends(require_api_key),
):
    """Save zone calibration polygons for a venue. Max 10 zones."""
    if len(zone_config.zones) > MAX_ZONES_PER_VENUE:
        raise HTTPException(status_code=400, detail=f"Max {MAX_ZONES_PER_VENUE} zones per venue")

    # Validate polygon points (minimum 3 vertices)
    for z in zone_config.zones:
        if len(z.points) < 3:
            raise HTTPException(status_code=400, detail=f"Zone '{z.name}' needs at least 3 points")

    row = await database.fetch_one(
        sqlalchemy.select(venues.c.config).where(venues.c.id == venue_id)
    )
    if not row:
        raise HTTPException(status_code=404, detail="Venue not found")

    existing_config = row["config"] or {}
    if isinstance(existing_config, str):
        existing_config = json.loads(existing_config)

    existing_config["zones_config"] = zone_config.dict()

    await database.execute(
        venues.update().where(venues.c.id == venue_id).values(
            config=json.dumps(existing_config)
        )
    )
    return success_response({"message": f"Saved {len(zone_config.zones)} zones", "zones": len(zone_config.zones)})
