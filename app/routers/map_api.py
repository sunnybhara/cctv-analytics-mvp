"""
Map API Endpoint
================
Venue data with visitor counts for map display.
"""

import sqlalchemy
from sqlalchemy import func
from fastapi import APIRouter
from app.responses import success_response

from app.database import database, venues, events

router = APIRouter()


@router.get("/api/map/venues")
async def get_map_venues():
    """Get all venues with their analytics for map display."""
    # Get all venues with location
    venue_query = sqlalchemy.select(
        venues.c.id, venues.c.name, venues.c.latitude, venues.c.longitude,
        venues.c.h3_zone, venues.c.city, venues.c.country, venues.c.venue_type
    )
    venue_rows = await database.fetch_all(venue_query)

    result = []
    for v in venue_rows:
        # Get visitor count for this venue
        count_query = sqlalchemy.select(
            func.count(func.distinct(events.c.pseudo_id))
        ).where(events.c.venue_id == v["id"])
        visitors = await database.fetch_val(count_query) or 0

        result.append({
            "id": v["id"],
            "name": v["name"],
            "latitude": v["latitude"],
            "longitude": v["longitude"],
            "h3_zone": v["h3_zone"],
            "city": v["city"],
            "country": v["country"],
            "venue_type": v["venue_type"],
            "visitors": visitors
        })

    return success_response({"venues": result})
