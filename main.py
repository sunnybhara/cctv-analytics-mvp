"""
Video Analytics MVP - Railway Backend
======================================
Receives events from edge devices, stores analytics, serves dashboard.

Deploy to Railway:
    railway login
    railway init
    railway up
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import databases
import sqlalchemy
from sqlalchemy import func

# =============================================================================
# DATABASE SETUP
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./analytics.db")

# Handle Railway's postgres:// vs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

database = databases.Database(DATABASE_URL)

metadata = sqlalchemy.MetaData()

# Events table - stores individual visitor events
events = sqlalchemy.Table(
    "events",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("venue_id", sqlalchemy.String(50), index=True),
    sqlalchemy.Column("pseudo_id", sqlalchemy.String(32), index=True),
    sqlalchemy.Column("timestamp", sqlalchemy.DateTime, index=True),
    sqlalchemy.Column("zone", sqlalchemy.String(50)),
    sqlalchemy.Column("dwell_seconds", sqlalchemy.Float, default=0),
    sqlalchemy.Column("age_bracket", sqlalchemy.String(10)),
    sqlalchemy.Column("gender", sqlalchemy.String(1)),
    sqlalchemy.Column("is_repeat", sqlalchemy.Boolean, default=False),
)

# Venues table
venues = sqlalchemy.Table(
    "venues",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String(50), primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(100)),
    sqlalchemy.Column("api_key", sqlalchemy.String(64)),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("config", sqlalchemy.JSON, default={}),
)

# Daily aggregates for fast queries
daily_stats = sqlalchemy.Table(
    "daily_stats",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("venue_id", sqlalchemy.String(50), index=True),
    sqlalchemy.Column("date", sqlalchemy.Date, index=True),
    sqlalchemy.Column("total_visitors", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("unique_visitors", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("repeat_visitors", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("avg_dwell_seconds", sqlalchemy.Float, default=0),
    sqlalchemy.Column("peak_hour", sqlalchemy.Integer),
    sqlalchemy.Column("gender_male", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("gender_female", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("age_20s", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("age_30s", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("age_40s", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("age_50plus", sqlalchemy.Integer, default=0),
)

engine = sqlalchemy.create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
metadata.create_all(engine)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

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
    """Create a new venue."""
    id: str
    name: str


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


# =============================================================================
# APP SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()

app = FastAPI(
    title="Video Analytics MVP",
    description="Privacy-preserving venue analytics API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple dashboard."""
    public_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "localhost:8000")
    base_url = f"https://{public_domain}" if "railway" in public_domain else f"http://{public_domain}"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Analytics MVP</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #333; }}
            .card {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .stat {{ font-size: 2em; font-weight: bold; color: #0066cc; }}
            .label {{ color: #666; font-size: 0.9em; }}
            .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
            pre {{ background: #1a1a1a; color: #0f0; padding: 15px; border-radius: 8px; overflow-x: auto; }}
            a {{ color: #0066cc; }}
        </style>
    </head>
    <body>
        <h1>Video Analytics MVP</h1>
        <p>Privacy-preserving venue analytics. Pseudonymized visitor tracking.</p>

        <div class="card">
            <h3>API Endpoints</h3>
            <ul>
                <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                <li><code>POST /events</code> - Submit batch events from edge device</li>
                <li><code>POST /events/batch</code> - Submit batch events (array format)</li>
                <li><code>GET /analytics/{{venue_id}}</code> - Get venue analytics</li>
                <li><code>GET /analytics/{{venue_id}}/hourly</code> - Hourly breakdown</li>
                <li><code>GET /venues</code> - List venues</li>
            </ul>
        </div>

        <div class="card">
            <h3>Quick Test</h3>
            <p>Submit test events:</p>
            <pre>
curl -X POST {base_url}/events \\
  -H "Content-Type: application/json" \\
  -d '{{
    "venue_id": "test_venue",
    "api_key": "test_key",
    "events": [
      {{
        "pseudo_id": "abc123",
        "timestamp": "2026-01-28T20:00:00",
        "zone": "bar",
        "dwell_seconds": 120,
        "age_bracket": "30s",
        "gender": "M"
      }}
    ]
  }}'
            </pre>
        </div>

        <div class="card">
            <h3>Status</h3>
            <p>API Running</p>
            <p>Database: {"PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite"}</p>
        </div>
    </body>
    </html>
    """


@app.post("/venues")
async def create_venue(venue: VenueCreate):
    """Register a new venue."""
    import secrets
    api_key = secrets.token_hex(32)

    query = venues.insert().values(
        id=venue.id,
        name=venue.name,
        api_key=api_key,
        created_at=datetime.utcnow()
    )

    try:
        await database.execute(query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Venue already exists: {e}")

    return {"venue_id": venue.id, "api_key": api_key, "message": "Save this API key - it won't be shown again"}


@app.get("/venues")
async def list_venues():
    """List all venues (without API keys)."""
    query = sqlalchemy.select(venues.c.id, venues.c.name, venues.c.created_at)
    rows = await database.fetch_all(query)
    return [{"id": r["id"], "name": r["name"], "created_at": r["created_at"]} for r in rows]


@app.post("/events")
async def submit_events(batch: EventBatch):
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

    return {"status": "ok", "inserted": inserted}


@app.post("/events/batch")
async def submit_events_batch(event_list: List[SingleEvent]):
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

    return {"status": "ok", "inserted": inserted}


@app.get("/stats/{venue_id}")
async def get_stats(
    venue_id: str,
    days: int = Query(default=7, ge=1, le=90)
) -> StatsResponse:
    """Get analytics for a venue (legacy endpoint, use /analytics/{venue_id})."""
    return await get_analytics(venue_id, days)


@app.get("/analytics/{venue_id}")
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

    return StatsResponse(
        venue_id=venue_id,
        period=f"Last {days} days",
        total_visitors=total,
        unique_visitors=unique,
        repeat_rate=round(repeats / total * 100, 1) if total > 0 else 0,
        avg_dwell_minutes=round(avg_dwell / 60, 1),
        peak_hour=peak_hour,
        gender_split=gender_split,
        age_distribution=age_dist
    )


@app.get("/analytics/{venue_id}/hourly")
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


@app.get("/stats/{venue_id}/hourly")
async def get_hourly_stats(
    venue_id: str,
    date: Optional[str] = None
):
    """Get hourly breakdown (legacy endpoint)."""
    return await get_hourly_analytics(venue_id, date)


@app.get("/health")
async def health():
    """Health check for Railway."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
