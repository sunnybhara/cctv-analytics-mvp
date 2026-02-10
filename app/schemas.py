"""
Pydantic Models
===============
Request/response schemas for API endpoints.
"""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class EventIn(BaseModel):
    """Single visitor event from edge device."""
    pseudo_id: str
    timestamp: datetime
    zone: str
    dwell_seconds: float = 0
    age_bracket: Optional[str] = None
    gender: Optional[str] = None
    is_repeat: bool = False


class EventBatch(BaseModel):
    """Batch of events from edge device."""
    venue_id: str
    api_key: str
    events: List[EventIn]


class SingleEvent(BaseModel):
    """Single event with venue info for direct ingestion."""
    venue_id: str
    pseudo_id: str
    timestamp: datetime
    zone: str
    dwell_seconds: float = 0
    age_bracket: Optional[str] = None
    gender: Optional[str] = None
    is_repeat: bool = False


class VenueCreate(BaseModel):
    """Create a new venue with optional geo-location."""
    id: str
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    venue_type: Optional[str] = None


class StatsResponse(BaseModel):
    """Analytics response."""
    venue_id: str
    period: str
    total_visitors: int
    unique_visitors: int
    repeat_rate: float
    avg_dwell_minutes: float
    peak_hour: Optional[int]
    gender_split: dict
    age_distribution: dict
    hourly_breakdown: Optional[list] = None
    confidence_level: Optional[float] = None
    visitor_range: Optional[dict] = None
    data_quality: Optional[str] = None
