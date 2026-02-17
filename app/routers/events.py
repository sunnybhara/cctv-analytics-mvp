"""
Event Ingestion Endpoints
=========================
Receive events from edge devices and provide legacy stats endpoint.
"""

from typing import List

from fastapi import APIRouter, Query, Depends, Request
from app.auth import require_api_key, verify_venue_access

from app.database import database, events
from app.responses import success_response
from app.schemas import EventBatch, SingleEvent
from app import limiter

router = APIRouter()


@limiter.limit("100/minute")
@router.post("/events")
async def submit_events(request: Request, batch: EventBatch, auth_venue_id: str = Depends(require_api_key)):
    """
    Receive batch events from edge device.
    Uses authenticated venue_id for inserts (prevents cross-venue injection).
    """
    # In production, use the venue_id from the API key (not from the body)
    venue_id = batch.venue_id if auth_venue_id == "__dev__" else auth_venue_id

    inserted = 0
    for event in batch.events:
        query = events.insert().values(
            venue_id=venue_id,
            pseudo_id=event.pseudo_id,
            timestamp=event.timestamp,
            zone=event.zone,
            dwell_seconds=event.dwell_seconds,
            age_bracket=event.age_bracket,
            gender=event.gender,
            is_repeat=event.is_repeat
        )
        await database.execute(query)
        inserted += 1

    return success_response({"status": "ok", "inserted": inserted})


@limiter.limit("100/minute")
@router.post("/events/batch")
async def submit_events_batch(request: Request, event_list: List[SingleEvent], auth_venue_id: str = Depends(require_api_key)):
    """
    Receive batch events as a simple array.
    Alternative format for edge devices.
    """
    inserted = 0
    for event in event_list:
        # In production, override venue_id from auth to prevent cross-venue injection
        venue_id = event.venue_id if auth_venue_id == "__dev__" else auth_venue_id
        query = events.insert().values(
            venue_id=venue_id,
            pseudo_id=event.pseudo_id,
            timestamp=event.timestamp,
            zone=event.zone,
            dwell_seconds=event.dwell_seconds,
            age_bracket=event.age_bracket,
            gender=event.gender,
            is_repeat=event.is_repeat
        )
        await database.execute(query)
        inserted += 1

    return success_response({"status": "ok", "inserted": inserted})


@router.get("/stats/{venue_id}")
async def get_stats(
    venue_id: str,
    days: int = Query(default=7, ge=1, le=90),
    auth_venue_id: str = Depends(require_api_key)
):
    """Get analytics for a venue (legacy endpoint, use /analytics/{venue_id})."""
    verify_venue_access(auth_venue_id, venue_id)
    from app.routers.analytics import get_analytics
    return await get_analytics(venue_id, days)
