"""
Return Visitor Analytics (ReID) Endpoints
==========================================
Known visitors, loyalty scoring, visit history, and returning visitor analytics.
"""

from datetime import datetime, timedelta

import sqlalchemy
from fastapi import Depends, APIRouter, HTTPException, Query
from app.auth import require_api_key
from app.responses import success_response

from app.database import database, events, visitor_embeddings

router = APIRouter()


def calculate_loyalty_score(visitor_row) -> str:
    """Calculate a loyalty tier based on visit count and recency."""
    visits = visitor_row["visit_count"] or 1
    last_seen = visitor_row["last_seen"]

    if last_seen:
        days_since = (datetime.utcnow() - last_seen).days
    else:
        days_since = 999

    # Loyalty tiers
    if visits >= 10 and days_since <= 30:
        return "VIP"
    elif visits >= 5 and days_since <= 30:
        return "Regular"
    elif visits >= 2 and days_since <= 60:
        return "Returning"
    elif days_since > 90:
        return "Lapsed"
    else:
        return "New"


@router.get("/api/visitors/{venue_id}")
async def get_known_visitors(
    venue_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    sort_by: str = "last_seen",  # last_seen, first_seen, visit_count
    _api_key: str = Depends(require_api_key)
):
    """
    Get all known visitors for a venue.
    These are visitors with face embeddings that can be tracked across sessions.
    """
    from sqlalchemy import func

    # Get total count
    total = await database.fetch_val(
        sqlalchemy.select(func.count()).select_from(visitor_embeddings).where(
            visitor_embeddings.c.venue_id == venue_id
        )
    ) or 0

    # Build sort order
    if sort_by == "first_seen":
        order = visitor_embeddings.c.first_seen.desc()
    elif sort_by == "visit_count":
        order = visitor_embeddings.c.visit_count.desc()
    else:
        order = visitor_embeddings.c.last_seen.desc()

    query = sqlalchemy.select(
        visitor_embeddings.c.visitor_id,
        visitor_embeddings.c.first_seen,
        visitor_embeddings.c.last_seen,
        visitor_embeddings.c.visit_count,
        visitor_embeddings.c.total_dwell_seconds,
        visitor_embeddings.c.age_bracket,
        visitor_embeddings.c.gender,
        visitor_embeddings.c.quality_score
    ).where(
        visitor_embeddings.c.venue_id == venue_id
    ).order_by(order).limit(limit).offset(offset)

    rows = await database.fetch_all(query)

    visitors = []
    for row in rows:
        visitors.append({
            "visitor_id": row["visitor_id"],
            "first_seen": row["first_seen"].isoformat() if row["first_seen"] else None,
            "last_seen": row["last_seen"].isoformat() if row["last_seen"] else None,
            "visit_count": row["visit_count"],
            "total_dwell_minutes": round((row["total_dwell_seconds"] or 0) / 60, 1),
            "avg_dwell_minutes": round((row["total_dwell_seconds"] or 0) / max(row["visit_count"], 1) / 60, 1),
            "age_bracket": row["age_bracket"],
            "gender": row["gender"],
            "loyalty_score": calculate_loyalty_score(row)
        })

    return success_response({
        "venue_id": venue_id,
        "total_known_visitors": total,
        "visitors": visitors
    }, pagination={
        "limit": limit, "offset": offset, "total": total,
        "has_more": offset + limit < total
    })


@router.get("/api/visitors/{venue_id}/stats")
async def get_visitor_loyalty_stats(venue_id: str, _api_key: str = Depends(require_api_key)):
    """
    Get loyalty statistics for a venue.
    Shows distribution of visitor loyalty tiers.
    """
    query = sqlalchemy.select(visitor_embeddings).where(
        visitor_embeddings.c.venue_id == venue_id
    )
    rows = await database.fetch_all(query)

    if not rows:
        return success_response({
            "venue_id": venue_id,
            "total_known_visitors": 0,
            "loyalty_distribution": {},
            "avg_visits_per_visitor": 0,
            "avg_lifetime_dwell_minutes": 0
        })

    # Calculate stats
    loyalty_counts = {"VIP": 0, "Regular": 0, "Returning": 0, "New": 0, "Lapsed": 0}
    total_visits = 0
    total_dwell = 0

    for row in rows:
        tier = calculate_loyalty_score(row)
        loyalty_counts[tier] += 1
        total_visits += row["visit_count"] or 1
        total_dwell += row["total_dwell_seconds"] or 0

    total_visitors = len(rows)

    return success_response({
        "venue_id": venue_id,
        "total_known_visitors": total_visitors,
        "loyalty_distribution": loyalty_counts,
        "loyalty_percentages": {
            k: round(v / total_visitors * 100, 1) for k, v in loyalty_counts.items() if v > 0
        },
        "avg_visits_per_visitor": round(total_visits / total_visitors, 1),
        "avg_lifetime_dwell_minutes": round(total_dwell / total_visitors / 60, 1),
        "total_return_visits": total_visits - total_visitors  # Total visits minus first visits
    })


@router.get("/api/visitors/{venue_id}/history/{visitor_id}")
async def get_visitor_history(venue_id: str, visitor_id: str, _api_key: str = Depends(require_api_key)):
    """
    Get visit history for a specific visitor.
    Shows all sessions and events for this person.
    """
    # Get visitor info
    visitor_query = sqlalchemy.select(visitor_embeddings).where(
        visitor_embeddings.c.venue_id == venue_id,
        visitor_embeddings.c.visitor_id == visitor_id
    )
    visitor = await database.fetch_one(visitor_query)

    if not visitor:
        raise HTTPException(status_code=404, detail="Visitor not found")

    # Get all events for this visitor
    events_query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.pseudo_id == visitor_id
    ).order_by(events.c.timestamp.desc()).limit(100)

    event_rows = await database.fetch_all(events_query)

    # Group events by date (session)
    sessions = {}
    for row in event_rows:
        date_key = row["timestamp"].strftime("%Y-%m-%d")
        if date_key not in sessions:
            sessions[date_key] = {
                "date": date_key,
                "events": [],
                "total_dwell": 0,
                "zones": set()
            }
        sessions[date_key]["events"].append({
            "timestamp": row["timestamp"].isoformat(),
            "zone": row["zone"],
            "dwell_seconds": row["dwell_seconds"]
        })
        sessions[date_key]["total_dwell"] += row["dwell_seconds"] or 0
        sessions[date_key]["zones"].add(row["zone"])

    # Convert to list
    session_list = []
    for date_key in sorted(sessions.keys(), reverse=True):
        s = sessions[date_key]
        session_list.append({
            "date": s["date"],
            "event_count": len(s["events"]),
            "total_dwell_minutes": round(s["total_dwell"] / 60, 1),
            "zones_visited": list(s["zones"])
        })

    return success_response({
        "venue_id": venue_id,
        "visitor_id": visitor_id,
        "profile": {
            "first_seen": visitor["first_seen"].isoformat() if visitor["first_seen"] else None,
            "last_seen": visitor["last_seen"].isoformat() if visitor["last_seen"] else None,
            "visit_count": visitor["visit_count"],
            "total_dwell_minutes": round((visitor["total_dwell_seconds"] or 0) / 60, 1),
            "age_bracket": visitor["age_bracket"],
            "gender": visitor["gender"],
            "loyalty_tier": calculate_loyalty_score(visitor)
        },
        "sessions": session_list
    })


@router.get("/api/visitors/{venue_id}/returning")
async def get_returning_visitor_analytics(
    venue_id: str,
    days: int = 30,
    _api_key: str = Depends(require_api_key)
):
    """
    Get analytics specifically about returning visitors.
    Useful for measuring loyalty and retention.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Get all events in period
    events_query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    event_rows = await database.fetch_all(events_query)

    # Get known visitors
    visitors_query = sqlalchemy.select(visitor_embeddings).where(
        visitor_embeddings.c.venue_id == venue_id
    )
    visitor_rows = await database.fetch_all(visitors_query)

    # Create lookup of known visitors
    known_visitor_ids = {r["visitor_id"] for r in visitor_rows}
    known_visitors_map = {r["visitor_id"]: r for r in visitor_rows}

    # Analyze events
    total_events = len(event_rows)
    unique_visitors = set()
    known_visitors_seen = set()
    return_events = 0

    for row in event_rows:
        unique_visitors.add(row["pseudo_id"])
        if row["pseudo_id"] in known_visitor_ids:
            known_visitors_seen.add(row["pseudo_id"])
        if row["is_repeat"]:
            return_events += 1

    # Calculate retention (visitors who came back)
    returning_visitors = [
        v for v in visitor_rows
        if (v["visit_count"] or 1) > 1 and v["visitor_id"] in known_visitors_seen
    ]

    return success_response({
        "venue_id": venue_id,
        "period": f"Last {days} days",
        "summary": {
            "total_visitors": len(unique_visitors),
            "known_visitors": len(known_visitors_seen),
            "tracking_rate": round(len(known_visitors_seen) / len(unique_visitors) * 100, 1) if unique_visitors else 0,
            "return_events": return_events,
            "return_event_rate": round(return_events / total_events * 100, 1) if total_events > 0 else 0
        },
        "retention": {
            "visitors_with_multiple_visits": len(returning_visitors),
            "retention_rate": round(len(returning_visitors) / len(known_visitors_seen) * 100, 1) if known_visitors_seen else 0
        },
        "top_returning_visitors": [
            {
                "visitor_id": v["visitor_id"],
                "visit_count": v["visit_count"],
                "last_seen": v["last_seen"].isoformat() if v["last_seen"] else None,
                "loyalty_tier": calculate_loyalty_score(v)
            }
            for v in sorted(returning_visitors, key=lambda x: x["visit_count"] or 0, reverse=True)[:10]
        ]
    })
