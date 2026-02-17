"""
Map API Endpoint
================
Venue data with visitor counts for map display.
"""

import sqlalchemy
from sqlalchemy import func
from fastapi import APIRouter, Depends
from app.auth import require_api_key
from app.responses import success_response

from app.database import database, venues, events, cohorts, cohort_members

router = APIRouter()


@router.get("/api/map/venues")
async def get_map_venues(_api_key: str = Depends(require_api_key)):
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

        # Get cohorts this venue belongs to
        cohort_query = sqlalchemy.select(
            cohorts.c.id, cohorts.c.name, cohorts.c.color
        ).select_from(
            cohort_members.join(cohorts, cohort_members.c.cohort_id == cohorts.c.id)
        ).where(cohort_members.c.venue_id == v["id"])
        venue_cohorts = await database.fetch_all(cohort_query)

        result.append({
            "id": v["id"],
            "name": v["name"],
            "latitude": v["latitude"],
            "longitude": v["longitude"],
            "h3_zone": v["h3_zone"],
            "city": v["city"],
            "country": v["country"],
            "venue_type": v["venue_type"],
            "visitors": visitors,
            "cohorts": [{"id": c["id"], "name": c["name"], "color": c["color"]} for c in venue_cohorts]
        })

    # Also return all cohorts for the filter UI
    all_cohorts = await database.fetch_all(
        sqlalchemy.select(cohorts.c.id, cohorts.c.name, cohorts.c.color)
    )

    return success_response({
        "venues": result,
        "cohorts": [{"id": c["id"], "name": c["name"], "color": c["color"]} for c in all_cohorts]
    })
