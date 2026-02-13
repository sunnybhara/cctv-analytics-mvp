"""
Nielsen-Style Analytics Endpoints (Phase 5)
============================================
Demographics, zones, trends, executive summary, heatmap, and export endpoints.
"""

from datetime import datetime, timedelta

import sqlalchemy
from fastapi import APIRouter, Depends
from app.auth import require_api_key
from fastapi.responses import HTMLResponse

from app.database import database, events

router = APIRouter()


@router.get("/analytics/{venue_id}/demographics")
async def get_demographics_analytics(
    venue_id: str,
    days: int = 7,
    _api_key: str = Depends(require_api_key)
):
    """
    Get demographic breakdown (Nielsen-style).
    Returns age and gender distribution with week-over-week comparison.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    prev_start = start_date - timedelta(days=days)

    # Current period
    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    current_rows = await database.fetch_all(query)

    # Previous period for comparison
    query_prev = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= prev_start,
        events.c.timestamp < start_date
    )
    prev_rows = await database.fetch_all(query_prev)

    def calc_demographics(rows):
        total = len(rows)
        if total == 0:
            return {"gender": {}, "age": {}, "total": 0}

        gender_counts = {"M": 0, "F": 0, "unknown": 0}
        age_counts = {"20s": 0, "30s": 0, "40s": 0, "50+": 0, "unknown": 0}

        for row in rows:
            g = row["gender"]
            a = row["age_bracket"]
            if g in gender_counts:
                gender_counts[g] += 1
            else:
                gender_counts["unknown"] += 1
            if a in age_counts:
                age_counts[a] += 1
            else:
                age_counts["unknown"] += 1

        return {
            "gender": {k: round(v / total * 100, 1) for k, v in gender_counts.items() if v > 0},
            "age": {k: round(v / total * 100, 1) for k, v in age_counts.items() if v > 0},
            "total": total
        }

    current = calc_demographics(current_rows)
    previous = calc_demographics(prev_rows)

    # Calculate changes
    gender_change = {}
    for k in current["gender"]:
        prev_val = previous["gender"].get(k, 0)
        gender_change[k] = round(current["gender"][k] - prev_val, 1)

    age_change = {}
    for k in current["age"]:
        prev_val = previous["age"].get(k, 0)
        age_change[k] = round(current["age"][k] - prev_val, 1)

    return {
        "venue_id": venue_id,
        "period": f"Last {days} days",
        "current": current,
        "previous": previous,
        "change": {
            "gender": gender_change,
            "age": age_change,
            "total_visitors": current["total"] - previous["total"]
        }
    }


@router.get("/analytics/{venue_id}/zones")
async def get_zone_analytics(
    venue_id: str,
    days: int = 7,
    _api_key: str = Depends(require_api_key)
):
    """
    Get zone performance analytics (Nielsen-style).
    Shows traffic, dwell time, and engagement per zone.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    rows = await database.fetch_all(query)

    # Aggregate by zone
    zones = {}
    total_visitors = set()

    for row in rows:
        zone = row["zone"] or "unknown"
        if zone not in zones:
            zones[zone] = {
                "visitors": set(),
                "total_dwell": 0,
                "event_count": 0
            }
        zones[zone]["visitors"].add(row["pseudo_id"])
        zones[zone]["total_dwell"] += row["dwell_seconds"] or 0
        zones[zone]["event_count"] += 1
        total_visitors.add(row["pseudo_id"])

    total_count = len(total_visitors) or 1  # Avoid division by zero

    result = []
    for zone, data in sorted(zones.items(), key=lambda x: len(x[1]["visitors"]), reverse=True):
        visitor_count = len(data["visitors"])
        avg_dwell = data["total_dwell"] / data["event_count"] if data["event_count"] > 0 else 0

        result.append({
            "zone": zone,
            "visitors": visitor_count,
            "traffic_percent": round(visitor_count / total_count * 100, 1),
            "avg_dwell_minutes": round(avg_dwell / 60, 1),
            "engagement": "High" if avg_dwell > 300 else "Medium" if avg_dwell > 120 else "Low"
        })

    return {
        "venue_id": venue_id,
        "period": f"Last {days} days",
        "total_unique_visitors": len(total_visitors),
        "zones": result
    }


@router.get("/analytics/{venue_id}/trends")
async def get_trend_analytics(
    venue_id: str,
    weeks: int = 8,
    _api_key: str = Depends(require_api_key)
):
    """
    Get weekly trends (Nielsen-style).
    Shows visitor counts, dwell time, and demographics over time.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(weeks=weeks)

    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    ).order_by(events.c.timestamp)

    rows = await database.fetch_all(query)

    # Group by week
    weekly = {}
    for row in rows:
        # Get ISO week
        week_start = row["timestamp"] - timedelta(days=row["timestamp"].weekday())
        week_key = week_start.strftime("%Y-%m-%d")

        if week_key not in weekly:
            weekly[week_key] = {
                "visitors": set(),
                "total_dwell": 0,
                "event_count": 0,
                "return_count": 0
            }

        weekly[week_key]["visitors"].add(row["pseudo_id"])
        weekly[week_key]["total_dwell"] += row["dwell_seconds"] or 0
        weekly[week_key]["event_count"] += 1
        if row["is_repeat"]:
            weekly[week_key]["return_count"] += 1

    # Convert to list
    result = []
    for week_key in sorted(weekly.keys()):
        data = weekly[week_key]
        visitor_count = len(data["visitors"])
        avg_dwell = data["total_dwell"] / data["event_count"] if data["event_count"] > 0 else 0
        return_rate = data["return_count"] / visitor_count * 100 if visitor_count > 0 else 0

        result.append({
            "week": week_key,
            "visitors": visitor_count,
            "avg_dwell_minutes": round(avg_dwell / 60, 1),
            "return_rate_percent": round(return_rate, 1)
        })

    # Calculate trend
    if len(result) >= 2:
        first_half = sum(r["visitors"] for r in result[:len(result)//2])
        second_half = sum(r["visitors"] for r in result[len(result)//2:])
        trend = "growing" if second_half > first_half else "declining" if second_half < first_half else "stable"
        growth_rate = round((second_half - first_half) / (first_half or 1) * 100, 1)
    else:
        trend = "insufficient_data"
        growth_rate = 0

    return {
        "venue_id": venue_id,
        "period": f"Last {weeks} weeks",
        "weekly": result,
        "trend": trend,
        "growth_rate_percent": growth_rate
    }


@router.get("/analytics/{venue_id}/summary")
async def get_executive_summary(
    venue_id: str,
    days: int = 7,
    _api_key: str = Depends(require_api_key)
):
    """
    Executive summary (Nielsen-style).
    Key metrics and insights for quick overview.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    prev_start = start_date - timedelta(days=days)

    # Current period
    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    current_rows = await database.fetch_all(query)

    # Previous period
    query_prev = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= prev_start,
        events.c.timestamp < start_date
    )
    prev_rows = await database.fetch_all(query_prev)

    def calc_metrics(rows):
        if not rows:
            return {
                "total_visitors": 0,
                "unique_visitors": 0,
                "return_visitors": 0,
                "return_rate_percent": 0,
                "avg_dwell_minutes": 0,
                "peak_hour": None,
                "peak_day": None,
                "top_demographic": None,
                "avg_engagement": None,
                "engaged_percent": None,
            }

        visitors = set()
        return_count = 0
        total_dwell = 0
        hourly = {}
        daily = {}
        demographics = {}
        engagement_scores = []
        engaged_count = 0

        for row in rows:
            visitors.add(row["pseudo_id"])
            if row["is_repeat"]:
                return_count += 1
            total_dwell += row["dwell_seconds"] or 0

            hour = row["timestamp"].hour
            day = row["timestamp"].strftime("%A")
            hourly[hour] = hourly.get(hour, 0) + 1
            daily[day] = daily.get(day, 0) + 1

            demo_key = f"{row['gender'] or '?'}/{row['age_bracket'] or '?'}"
            demographics[demo_key] = demographics.get(demo_key, 0) + 1

            # Engagement tracking
            engagement_val = row["engagement_score"]
            if engagement_val is not None:
                engagement_scores.append(engagement_val)
                if engagement_val >= 70:
                    engaged_count += 1

        unique = len(visitors)
        peak_hour = max(hourly, key=hourly.get) if hourly else None
        peak_day = max(daily, key=daily.get) if daily else None
        top_demo = max(demographics, key=demographics.get) if demographics else None

        # Calculate engagement metrics
        avg_engagement = round(sum(engagement_scores) / len(engagement_scores), 1) if engagement_scores else None
        engaged_percent = round(engaged_count / len(engagement_scores) * 100, 1) if engagement_scores else None

        return {
            "total_visitors": len(rows),
            "unique_visitors": unique,
            "return_visitors": return_count,
            "return_rate_percent": round(return_count / unique * 100, 1) if unique > 0 else 0,
            "avg_dwell_minutes": round(total_dwell / len(rows) / 60, 1) if rows else 0,
            "peak_hour": peak_hour,
            "peak_day": peak_day,
            "top_demographic": top_demo,
            "avg_engagement": avg_engagement,
            "engaged_percent": engaged_percent,
        }

    current = calc_metrics(current_rows)
    previous = calc_metrics(prev_rows)

    # Generate insights
    insights = []
    if current["unique_visitors"] > previous["unique_visitors"]:
        pct = round((current["unique_visitors"] - previous["unique_visitors"]) / (previous["unique_visitors"] or 1) * 100, 1)
        insights.append(f"Visitor traffic up {pct}% vs previous period")
    elif current["unique_visitors"] < previous["unique_visitors"]:
        pct = round((previous["unique_visitors"] - current["unique_visitors"]) / (previous["unique_visitors"] or 1) * 100, 1)
        insights.append(f"Visitor traffic down {pct}% vs previous period")

    if current["avg_dwell_minutes"] > previous["avg_dwell_minutes"]:
        insights.append(f"Dwell time improved to {current['avg_dwell_minutes']} min average")

    if current["return_rate_percent"] > 20:
        insights.append(f"Strong return rate at {current['return_rate_percent']}%")

    if current["peak_hour"]:
        insights.append(f"Peak traffic at {current['peak_hour']}:00")

    # Engagement insights
    if current["avg_engagement"] is not None:
        if current["avg_engagement"] >= 70:
            insights.append(f"High engagement score: {current['avg_engagement']}")
        elif current["avg_engagement"] >= 50:
            insights.append(f"Moderate engagement score: {current['avg_engagement']}")
        if current["engaged_percent"] and current["engaged_percent"] >= 30:
            insights.append(f"{current['engaged_percent']}% of visitors highly engaged")

    return {
        "venue_id": venue_id,
        "period": f"Last {days} days",
        "current": current,
        "previous": previous,
        "change": {
            "visitors": current["unique_visitors"] - previous["unique_visitors"],
            "visitors_percent": round((current["unique_visitors"] - previous["unique_visitors"]) / (previous["unique_visitors"] or 1) * 100, 1),
            "dwell_minutes": round(current["avg_dwell_minutes"] - previous["avg_dwell_minutes"], 1)
        },
        "insights": insights
    }


@router.get("/analytics/{venue_id}/heatmap")
async def get_hourly_heatmap(
    venue_id: str,
    weeks: int = 4,
    _api_key: str = Depends(require_api_key)
):
    """
    Get hourly heatmap data (Nielsen-style).
    Returns visitor counts by day-of-week and hour for heatmap visualization.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(weeks=weeks)

    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    rows = await database.fetch_all(query)

    # Build heatmap grid: day x hour
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap = {day: {h: 0 for h in range(24)} for day in days}

    for row in rows:
        day = row["timestamp"].strftime("%A")
        hour = row["timestamp"].hour
        if day in heatmap:
            heatmap[day][hour] += 1

    # Find max for normalization
    max_val = max(max(h.values()) for h in heatmap.values()) or 1

    # Convert to list format for frontend
    result = []
    for day in days:
        for hour in range(24):
            count = heatmap[day][hour]
            result.append({
                "day": day,
                "hour": hour,
                "count": count,
                "intensity": round(count / max_val, 2)  # 0-1 normalized
            })

    # Find peak times
    peak_times = sorted(result, key=lambda x: x["count"], reverse=True)[:5]

    return {
        "venue_id": venue_id,
        "period": f"Last {weeks} weeks",
        "heatmap": result,
        "peak_times": peak_times,
        "max_count": max_val
    }


@router.get("/analytics/{venue_id}/export")
async def export_analytics(
    venue_id: str,
    days: int = 7,
    format: str = "json",
    _api_key: str = Depends(require_api_key)
):
    """
    Export analytics data for reporting.
    Combines all analytics into a single exportable format.
    """
    from app.routers.behavior import get_behavior_analytics

    # Gather all analytics
    summary = await get_executive_summary(venue_id, days)
    demographics = await get_demographics_analytics(venue_id, days)
    zones = await get_zone_analytics(venue_id, days)
    heatmap = await get_hourly_heatmap(venue_id, weeks=max(1, days // 7))
    behavior = await get_behavior_analytics(venue_id, days)

    export_data = {
        "venue_id": venue_id,
        "generated_at": datetime.utcnow().isoformat(),
        "period_days": days,
        "executive_summary": summary,
        "demographics": demographics,
        "zone_performance": zones,
        "hourly_heatmap": {
            "peak_times": heatmap["peak_times"],
            "max_hourly_count": heatmap["max_count"]
        },
        "behavior_analytics": behavior
    }

    if format == "csv":
        # Return as downloadable CSV summary
        import io
        output = io.StringIO()
        output.write("CCTV Analytics Report\n")
        output.write(f"Venue: {venue_id}\n")
        output.write(f"Period: Last {days} days\n")
        output.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")

        output.write("KEY METRICS\n")
        output.write(f"Unique Visitors,{summary['current']['unique_visitors']}\n")
        output.write(f"Return Rate,{summary['current']['return_rate_percent']}%\n")
        output.write(f"Avg Dwell Time,{summary['current']['avg_dwell_minutes']} min\n")
        output.write(f"Peak Hour,{summary['current']['peak_hour']}:00\n")
        if summary['current'].get('avg_engagement'):
            output.write(f"Avg Engagement Score,{summary['current']['avg_engagement']}\n")
            output.write(f"Highly Engaged %,{summary['current']['engaged_percent']}%\n")

        return HTMLResponse(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={venue_id}_analytics.csv"}
        )

    return export_data
