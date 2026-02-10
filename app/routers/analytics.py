"""
Analytics Endpoints
===================
Core analytics: GET /analytics/{venue_id}, hourly breakdowns, and legacy stats endpoints.
"""

from datetime import datetime, timedelta
from typing import Optional

import sqlalchemy
from sqlalchemy import func
from fastapi import APIRouter, Query

from app.database import database, events
from app.config import DATABASE_URL
from app.schemas import StatsResponse

router = APIRouter()


@router.get("/analytics/{venue_id}")
async def get_analytics(
    venue_id: str,
    days: int = Query(default=7, ge=1, le=90)
) -> StatsResponse:
    """Get analytics for a venue."""

    since = datetime.utcnow() - timedelta(days=days)

    # Total events
    query = sqlalchemy.select(func.count(events.c.id)).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= since
    )
    total = await database.fetch_val(query) or 0

    # Unique visitors
    query = sqlalchemy.select(func.count(func.distinct(events.c.pseudo_id))).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= since
    )
    unique = await database.fetch_val(query) or 0

    # Repeat visitors
    query = sqlalchemy.select(func.count(events.c.id)).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= since,
        events.c.is_repeat == True
    )
    repeats = await database.fetch_val(query) or 0

    # Average dwell time
    query = sqlalchemy.select(func.avg(events.c.dwell_seconds)).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= since,
        events.c.dwell_seconds > 0
    )
    avg_dwell = await database.fetch_val(query) or 0

    # Gender split
    query = sqlalchemy.select(
        events.c.gender,
        func.count(events.c.id)
    ).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= since,
        events.c.gender.isnot(None)
    ).group_by(events.c.gender)

    gender_rows = await database.fetch_all(query)
    gender_split = {r["gender"]: r[1] for r in gender_rows}

    # Age distribution
    query = sqlalchemy.select(
        events.c.age_bracket,
        func.count(events.c.id)
    ).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= since,
        events.c.age_bracket.isnot(None)
    ).group_by(events.c.age_bracket)

    age_rows = await database.fetch_all(query)
    age_dist = {r["age_bracket"]: r[1] for r in age_rows}

    # Peak hour calculation
    peak_hour = None
    if total > 0:
        # For SQLite, extract hour differently
        if "sqlite" in DATABASE_URL:
            query = sqlalchemy.select(
                func.strftime('%H', events.c.timestamp).label('hour'),
                func.count(events.c.id).label('count')
            ).where(
                events.c.venue_id == venue_id,
                events.c.timestamp >= since
            ).group_by('hour').order_by(func.count(events.c.id).desc()).limit(1)
        else:
            query = sqlalchemy.select(
                sqlalchemy.extract('hour', events.c.timestamp).label('hour'),
                func.count(events.c.id).label('count')
            ).where(
                events.c.venue_id == venue_id,
                events.c.timestamp >= since
            ).group_by('hour').order_by(func.count(events.c.id).desc()).limit(1)

        peak_row = await database.fetch_one(query)
        if peak_row:
            peak_hour = int(peak_row["hour"])

    # Calculate confidence metrics from track quality data
    confidence_level = None
    visitor_range = None
    data_quality = None

    if unique > 0:
        # Get average track frames and detection confidence
        query = sqlalchemy.select(
            func.avg(events.c.track_frames).label('avg_frames'),
            func.avg(events.c.detection_conf).label('avg_conf')
        ).where(
            events.c.venue_id == venue_id,
            events.c.timestamp >= since,
            events.c.is_repeat == False  # Only primary events
        )
        quality_row = await database.fetch_one(query)

        avg_frames = quality_row["avg_frames"] if quality_row and quality_row["avg_frames"] else 5
        avg_conf = quality_row["avg_conf"] if quality_row and quality_row["avg_conf"] else 0.7

        # Confidence calculation:
        # - Track frames: more frames = higher confidence (5-30 frames maps to 0.5-1.0)
        # - Detection confidence: higher = better (0.5-1.0 maps to 0.5-1.0)
        frame_score = min(1.0, max(0.5, (avg_frames - 5) / 25 + 0.5)) if avg_frames else 0.5
        conf_score = max(0.5, min(1.0, avg_conf)) if avg_conf else 0.5
        confidence_level = round((frame_score * 0.6 + conf_score * 0.4), 2)

        # Calculate confidence interval (rough 95% CI)
        # Error margin decreases with higher confidence and more samples
        error_margin = max(1, int(unique * (1 - confidence_level) * 0.5))
        visitor_range = {
            "low": max(0, unique - error_margin),
            "high": unique + error_margin
        }

        # Data quality label
        if confidence_level >= 0.85:
            data_quality = "high"
        elif confidence_level >= 0.70:
            data_quality = "medium"
        else:
            data_quality = "low"

    return StatsResponse(
        venue_id=venue_id,
        period=f"Last {days} days",
        total_visitors=total,
        unique_visitors=unique,
        repeat_rate=round(repeats / total * 100, 1) if total > 0 else 0,
        avg_dwell_minutes=round(avg_dwell / 60, 1),
        peak_hour=peak_hour,
        gender_split=gender_split,
        age_distribution=age_dist,
        confidence_level=confidence_level,
        visitor_range=visitor_range,
        data_quality=data_quality
    )


@router.get("/analytics/{venue_id}/hourly")
async def get_hourly_analytics(
    venue_id: str,
    date: Optional[str] = None
):
    """Get hourly breakdown for a specific date."""

    if date:
        target_date = datetime.fromisoformat(date).date()
    else:
        target_date = datetime.utcnow().date()

    start = datetime.combine(target_date, datetime.min.time())
    end = start + timedelta(days=1)

    # Get all events for the day
    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start,
        events.c.timestamp < end
    )

    rows = await database.fetch_all(query)

    # Aggregate by hour
    hourly = {h: {"count": 0, "unique": set()} for h in range(24)}

    for row in rows:
        hour = row["timestamp"].hour
        hourly[hour]["count"] += 1
        hourly[hour]["unique"].add(row["pseudo_id"])

    return {
        "venue_id": venue_id,
        "date": str(target_date),
        "hourly": [
            {"hour": h, "visitors": data["count"], "unique": len(data["unique"])}
            for h, data in hourly.items()
        ]
    }


@router.get("/stats/{venue_id}/hourly")
async def get_hourly_stats(
    venue_id: str,
    date: Optional[str] = None
):
    """Get hourly breakdown (legacy endpoint)."""
    return await get_hourly_analytics(venue_id, date)
