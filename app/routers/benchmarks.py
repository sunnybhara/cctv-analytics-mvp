"""
Benchmarking & Comparison Endpoints
====================================
Venue comparison and industry benchmarks.
"""

from datetime import datetime, timedelta

import sqlalchemy
from fastapi import APIRouter, HTTPException

from app.database import database, events, venues

router = APIRouter()


@router.get("/api/benchmark/venues")
async def compare_venues(venue_ids: str, days: int = 7):
    """
    Compare multiple venues side by side.
    Pass venue IDs as comma-separated string: ?venue_ids=venue1,venue2,venue3
    """
    ids = [v.strip() for v in venue_ids.split(",") if v.strip()]

    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 venue IDs to compare")
    if len(ids) > 10:
        raise HTTPException(status_code=400, detail="Max 10 venues for comparison")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    results = []
    for venue_id in ids:
        query = sqlalchemy.select(events).where(
            events.c.venue_id == venue_id,
            events.c.timestamp >= start_date,
            events.c.timestamp < end_date
        )
        rows = await database.fetch_all(query)

        if not rows:
            results.append({
                "venue_id": venue_id,
                "unique_visitors": 0,
                "avg_dwell_minutes": 0,
                "return_rate_percent": 0
            })
            continue

        visitors = set()
        return_count = 0
        total_dwell = 0

        for row in rows:
            visitors.add(row["pseudo_id"])
            if row["is_repeat"]:
                return_count += 1
            total_dwell += row["dwell_seconds"] or 0

        unique = len(visitors)
        results.append({
            "venue_id": venue_id,
            "unique_visitors": unique,
            "avg_dwell_minutes": round(total_dwell / len(rows) / 60, 1) if rows else 0,
            "return_rate_percent": round(return_count / unique * 100, 1) if unique > 0 else 0
        })

    # Sort by visitors descending
    results.sort(key=lambda x: x["unique_visitors"], reverse=True)

    # Calculate averages for "industry" comparison
    avg_visitors = sum(r["unique_visitors"] for r in results) / len(results) if results else 0
    avg_dwell = sum(r["avg_dwell_minutes"] for r in results) / len(results) if results else 0
    avg_return = sum(r["return_rate_percent"] for r in results) / len(results) if results else 0

    return {
        "period": f"Last {days} days",
        "venues": results,
        "averages": {
            "unique_visitors": round(avg_visitors, 1),
            "avg_dwell_minutes": round(avg_dwell, 1),
            "return_rate_percent": round(avg_return, 1)
        }
    }


@router.get("/api/benchmark/industry")
async def get_industry_benchmarks(venue_type: str = "bar", days: int = 7):
    """
    Get industry benchmarks based on venue type.
    Calculates averages across all venues of the same type.
    """
    # Get all venues of this type
    venue_query = sqlalchemy.select(venues.c.id).where(venues.c.venue_type == venue_type)
    venue_rows = await database.fetch_all(venue_query)
    venue_ids = [r["id"] for r in venue_rows]

    if not venue_ids:
        # Return generic benchmarks if no venues of this type
        return {
            "venue_type": venue_type,
            "sample_size": 0,
            "benchmarks": {
                "avg_daily_visitors": 50,
                "avg_dwell_minutes": 25,
                "return_rate_percent": 20,
                "peak_hour": 20
            },
            "note": "Generic industry estimates - no venue data available"
        }

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Aggregate across all venues
    all_visitors = []
    all_dwells = []
    all_returns = []
    hourly_counts = {}

    for venue_id in venue_ids:
        query = sqlalchemy.select(events).where(
            events.c.venue_id == venue_id,
            events.c.timestamp >= start_date,
            events.c.timestamp < end_date
        )
        rows = await database.fetch_all(query)

        if rows:
            visitors = len(set(r["pseudo_id"] for r in rows))
            all_visitors.append(visitors / days)  # Daily average

            total_dwell = sum(r["dwell_seconds"] or 0 for r in rows)
            all_dwells.append(total_dwell / len(rows) / 60 if rows else 0)

            return_count = sum(1 for r in rows if r["is_repeat"])
            all_returns.append(return_count / visitors * 100 if visitors > 0 else 0)

            for row in rows:
                h = row["timestamp"].hour
                hourly_counts[h] = hourly_counts.get(h, 0) + 1

    peak_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else 20

    return {
        "venue_type": venue_type,
        "sample_size": len(venue_ids),
        "period": f"Last {days} days",
        "benchmarks": {
            "avg_daily_visitors": round(sum(all_visitors) / len(all_visitors), 1) if all_visitors else 0,
            "avg_dwell_minutes": round(sum(all_dwells) / len(all_dwells), 1) if all_dwells else 0,
            "return_rate_percent": round(sum(all_returns) / len(all_returns), 1) if all_returns else 0,
            "peak_hour": peak_hour
        }
    }
