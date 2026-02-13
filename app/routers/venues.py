"""
Venue CRUD Endpoints
====================
Create and list venues with optional geo-location.
"""

import secrets
from datetime import datetime

import sqlalchemy
from fastapi import APIRouter, HTTPException

from app.database import database, venues
from app.responses import success_response
from app.schemas import VenueCreate
from app.video.helpers import lat_long_to_h3

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
async def list_venues():
    """List all venues with geo info (without API keys)."""
    query = sqlalchemy.select(
        venues.c.id, venues.c.name, venues.c.created_at,
        venues.c.latitude, venues.c.longitude, venues.c.h3_zone,
        venues.c.city, venues.c.country, venues.c.venue_type
    )
    rows = await database.fetch_all(query)
    return success_response([{
        "id": r["id"],
        "name": r["name"],
        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        "latitude": r["latitude"],
        "longitude": r["longitude"],
        "h3_zone": r["h3_zone"],
        "city": r["city"],
        "country": r["country"],
        "venue_type": r["venue_type"]
    } for r in rows])
