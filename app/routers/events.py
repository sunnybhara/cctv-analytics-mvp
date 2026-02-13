"""
Event Ingestion Endpoints
=========================
Receive events from edge devices and provide legacy stats endpoint.
"""

from typing import List

from fastapi import APIRouter, Query, Depends, Request
from app.auth import require_api_key

from app.database import database, events
from app.responses import success_response
from app.schemas import EventBatch, SingleEvent
from app import limiter

router = APIRouter()


@limiter.limit("100/minute")
@router.post("/events")
async def submit_events(request: Request, batch: EventBatch, _api_key: str = Depends(require_api_key)):
    """
    Receive batch events from edge device.
    This is the main ingestion endpoint.
    """
    # TODO: Validate API key in production
    # For MVP, accept all events

    inserted = 0
    for event in batch.events:
        query = events.insert().values(
            venue_id=batch.venue_id,
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
async def submit_events_batch(request: Request, event_list: List[SingleEvent], _api_key: str = Depends(require_api_key)):
    """
    Receive batch events as a simple array.
    Alternative format for edge devices.
    """
    inserted = 0
    for event in event_list:
        query = events.insert().values(
            venue_id=event.venue_id,
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
    _api_key: str = Depends(require_api_key)
):
    """Get analytics for a venue (legacy endpoint, use /analytics/{venue_id})."""
    from app.routers.analytics import get_analytics
    return await get_analytics(venue_id, days)
