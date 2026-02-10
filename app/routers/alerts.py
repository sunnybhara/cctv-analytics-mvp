"""
Alerts & Anomaly Detection Endpoints
=====================================
Anomaly checking logic and CRUD endpoints for alerts.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict

import sqlalchemy
from fastapi import APIRouter, HTTPException

from app.database import database, events, alerts

router = APIRouter()


async def check_anomalies(venue_id: str) -> List[Dict]:
    """
    Check for anomalies and generate alerts.
    Called after video processing completes.
    """
    detected_alerts = []

    # Get recent data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    baseline_start = start_date - timedelta(days=7)

    # Current week events
    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    current_rows = await database.fetch_all(query)

    # Previous week events (baseline)
    query_baseline = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= baseline_start,
        events.c.timestamp < start_date
    )
    baseline_rows = await database.fetch_all(query_baseline)

    if not baseline_rows:
        return []  # Not enough data for comparison

    current_visitors = len(set(r["pseudo_id"] for r in current_rows))
    baseline_visitors = len(set(r["pseudo_id"] for r in baseline_rows))

    # Check for significant traffic changes
    if baseline_visitors > 0:
        change_pct = (current_visitors - baseline_visitors) / baseline_visitors * 100

        if change_pct > 50:
            detected_alerts.append({
                "alert_type": "traffic_spike",
                "severity": "info",
                "title": "Traffic Spike Detected",
                "message": f"Visitor traffic up {change_pct:.0f}% vs previous week ({current_visitors} vs {baseline_visitors})",
                "data": {"current": current_visitors, "baseline": baseline_visitors, "change_pct": change_pct}
            })
        elif change_pct < -30:
            detected_alerts.append({
                "alert_type": "traffic_drop",
                "severity": "warning",
                "title": "Traffic Drop Detected",
                "message": f"Visitor traffic down {abs(change_pct):.0f}% vs previous week ({current_visitors} vs {baseline_visitors})",
                "data": {"current": current_visitors, "baseline": baseline_visitors, "change_pct": change_pct}
            })

    # Check for unusual hour activity (between midnight and 5am)
    unusual_hours = [r for r in current_rows if r["timestamp"].hour < 5]
    if len(unusual_hours) > 10:
        detected_alerts.append({
            "alert_type": "unusual_hours",
            "severity": "info",
            "title": "After-Hours Activity",
            "message": f"Detected {len(unusual_hours)} events between midnight and 5am",
            "data": {"count": len(unusual_hours)}
        })

    # Store alerts
    for alert_data in detected_alerts:
        insert_query = alerts.insert().values(
            venue_id=venue_id,
            alert_type=alert_data["alert_type"],
            severity=alert_data["severity"],
            title=alert_data["title"],
            message=alert_data["message"],
            data=alert_data["data"],
            created_at=datetime.utcnow()
        )
        await database.execute(insert_query)

    return detected_alerts


@router.get("/api/alerts")
async def list_alerts(
    venue_id: Optional[str] = None,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = 50
):
    """List alerts, optionally filtered."""
    query = sqlalchemy.select(alerts).order_by(alerts.c.created_at.desc()).limit(limit)

    if venue_id:
        query = query.where(alerts.c.venue_id == venue_id)
    if severity:
        query = query.where(alerts.c.severity == severity)
    if acknowledged is not None:
        query = query.where(alerts.c.acknowledged == acknowledged)

    rows = await database.fetch_all(query)

    return {
        "alerts": [
            {
                "id": r["id"],
                "venue_id": r["venue_id"],
                "alert_type": r["alert_type"],
                "severity": r["severity"],
                "title": r["title"],
                "message": r["message"],
                "data": r["data"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "acknowledged": r["acknowledged"]
            }
            for r in rows
        ]
    }


@router.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """Mark an alert as acknowledged."""
    query = alerts.update().where(alerts.c.id == alert_id).values(
        acknowledged=True,
        acknowledged_at=datetime.utcnow()
    )
    result = await database.execute(query)

    if result == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"message": "Alert acknowledged", "alert_id": alert_id}


@router.post("/api/alerts/check/{venue_id}")
async def trigger_anomaly_check(venue_id: str):
    """Manually trigger anomaly detection for a venue."""
    detected = await check_anomalies(venue_id)
    return {
        "venue_id": venue_id,
        "alerts_generated": len(detected),
        "alerts": detected
    }
