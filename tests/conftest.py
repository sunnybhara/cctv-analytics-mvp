"""
Pytest configuration and fixtures for CCTV Analytics tests.
"""
import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from httpx import AsyncClient, ASGITransport

# Set test database before importing main
os.environ["DATABASE_URL"] = "sqlite:///./test_analytics.db"

from main import app, database, metadata, events, venues
import sqlalchemy


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def setup_database():
    """Create test database and tables."""
    engine = sqlalchemy.create_engine(
        "sqlite:///./test_analytics.db",
        connect_args={"check_same_thread": False}
    )
    metadata.create_all(engine)
    await database.connect()
    yield
    await database.disconnect()
    # Cleanup
    if os.path.exists("test_analytics.db"):
        os.remove("test_analytics.db")


@pytest_asyncio.fixture
async def client(setup_database):
    """Async HTTP client for API testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def test_venue(client):
    """Create a test venue and return its details."""
    response = await client.post("/venues", json={
        "id": "test_venue",
        "name": "Test Venue",
        "latitude": -26.2041,
        "longitude": 28.0473
    })
    data = response.json()
    yield data
    # Cleanup handled by database teardown


@pytest_asyncio.fixture
async def test_venue_with_data(client, test_venue, setup_database):
    """Create a test venue with sample event data."""
    venue_id = "test_venue"

    # Insert sample events
    engine = sqlalchemy.create_engine(
        "sqlite:///./test_analytics.db",
        connect_args={"check_same_thread": False}
    )

    now = datetime.utcnow()
    sample_events = []

    # Create 50 sample events over the past 7 days
    for i in range(50):
        hours_ago = i * 3  # Spread over time
        timestamp = now - timedelta(hours=hours_ago)
        sample_events.append({
            "venue_id": venue_id,
            "pseudo_id": f"visitor_{i % 10}",  # 10 unique visitors
            "timestamp": timestamp,
            "zone": ["entrance", "bar", "seating", "exit"][i % 4],
            "dwell_seconds": 60 + (i * 10) % 300,
            "age_bracket": ["18-24", "25-34", "35-44", "45-54"][i % 4],
            "gender": "M" if i % 3 == 0 else "F",
            "is_repeat": i % 5 == 0,
            "track_frames": 10 + i,
            "detection_conf": 0.85 + (i % 10) * 0.01,
            "engagement_score": 50 + (i % 50),
            "behavior_type": ["engaged", "browsing", "waiting", "passing"][i % 4],
            "body_orientation": -0.5 + (i % 10) * 0.1,
            "posture": ["upright", "leaning_forward", "arms_crossed"][i % 3],
        })

    with engine.connect() as conn:
        for event in sample_events:
            conn.execute(events.insert().values(**event))
        conn.commit()

    yield {"venue_id": venue_id, "event_count": len(sample_events)}


@pytest.fixture
def sample_video_path():
    """Path to a sample test video (if available)."""
    # For real testing, you'd have a sample video file
    return None
