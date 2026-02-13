"""
Event Models + API Poster
==========================
BarEvent dataclass and async HTTP poster.
"""

import logging
import time
import uuid
from dataclasses import asdict, dataclass

import httpx

from config.settings import CAMERA_ID, EVENT_API_KEY, EVENT_API_URL

logger = logging.getLogger(__name__)


@dataclass
class BarEvent:
    event_type: str           # "serve", "payment", "queue_state", etc.
    venue_id: str
    camera_id: str
    zone: str
    confidence: float
    action: str               # from VLM
    drink_category: str = ""
    vessel_type: str = ""
    description: str = ""
    person_track_id: int = 0
    object_track_id: int = 0
    timestamp: float = 0.0
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()


async def post_events(events: list[BarEvent]) -> bool:
    """POST verified events to the backend API.

    Returns True on success, False on failure.
    """
    if not events:
        return True

    payload = {
        "venue_id": events[0].venue_id,
        "events": [asdict(e) for e in events],
    }

    headers = {}
    if EVENT_API_KEY:
        headers["X-API-Key"] = EVENT_API_KEY

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(EVENT_API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                logger.info(f"Posted {len(events)} events")
                return True
            else:
                logger.error(
                    f"Event API returned {response.status_code}: {response.text[:200]}"
                )
                return False
    except Exception as e:
        logger.error(f"Failed to post events: {e}")
        return False
