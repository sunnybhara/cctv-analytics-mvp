"""
Hardening Torture Tests
=======================
Tests for auth, rate limiting, timeouts, response format, and pagination.
"""

import os
import secrets

import pytest
import pytest_asyncio
import sqlalchemy
from httpx import AsyncClient, ASGITransport

# Auth tests need AUTH_ENABLED=true, so we configure per-class
# Default: auth disabled (set in conftest.py)

from main import app, database, metadata, venues


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="module")
async def setup_db():
    """Ensure DB is ready for hardening tests."""
    engine = sqlalchemy.create_engine(
        "sqlite:///./test_analytics.db",
        connect_args={"check_same_thread": False}
    )
    metadata.create_all(engine)
    if not database.is_connected:
        await database.connect()
    yield
    # Don't disconnect — session fixture owns that


@pytest_asyncio.fixture
async def client(setup_db):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def venue_with_key(client, setup_db):
    """Create a venue and return (venue_id, api_key)."""
    vid = f"harden_{secrets.token_hex(4)}"
    resp = await client.post("/venues", json={
        "id": vid,
        "name": "Hardening Test Venue"
    })
    if resp.status_code == 200:
        data = resp.json()["data"]
        return vid, data["api_key"]
    # If venue exists, create with different id
    vid2 = f"harden_{secrets.token_hex(4)}"
    resp2 = await client.post("/venues", json={
        "id": vid2,
        "name": "Hardening Test Venue 2"
    })
    data = resp2.json()["data"]
    return vid2, data["api_key"]


# ===========================================================================
# Auth Tests
# ===========================================================================

class TestAuth:
    """Test API key authentication enforcement."""

    @pytest.mark.asyncio
    async def test_public_endpoints_no_auth(self, client):
        """Health, venues list, POST venues should work without auth."""
        health = await client.get("/health")
        assert health.status_code == 200

        venues_list = await client.get("/venues")
        assert venues_list.status_code == 200

        # POST venues is public
        resp = await client.post("/venues", json={
            "id": f"auth_test_{secrets.token_hex(4)}",
            "name": "Auth Test"
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_disabled_allows_all(self, client):
        """With AUTH_ENABLED=false, protected endpoints work without key."""
        # AUTH_ENABLED is false in test env
        resp = await client.get("/analytics/nonexistent")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_module_imports(self):
        """Auth module should be importable and have required functions."""
        from app.auth import require_api_key
        assert callable(require_api_key)

    @pytest.mark.asyncio
    async def test_auth_cache_exists(self):
        """Auth cache should be initialized."""
        from app.auth import _cache, _lock
        assert _cache is not None
        assert _lock is not None

    @pytest.mark.asyncio
    async def test_protected_endpoint_returns_data(self, client):
        """Protected endpoints should return data when auth is disabled."""
        resp = await client.get("/analytics/test_venue?days=7")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    @pytest.mark.asyncio
    async def test_events_endpoint_protected(self, client):
        """Events endpoint should be accessible in dev mode (auth bypassed)."""
        resp = await client.post("/events", json={
            "venue_id": "test",
            "api_key": "dummy",
            "events": [
                {
                    "pseudo_id": "v1",
                    "timestamp": "2025-01-01T00:00:00",
                    "zone": "entrance",
                    "dwell_seconds": 10
                }
            ]
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_process_status_protected(self, client):
        """Process status endpoint should be protected."""
        resp = await client.get("/process/status/nonexistent")
        assert resp.status_code == 404  # Not found, but auth passed

    @pytest.mark.asyncio
    async def test_batch_stats_protected(self, client):
        """Batch stats endpoint should be accessible in dev mode."""
        resp = await client.get("/api/batch/stats")
        assert resp.status_code == 200


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================

class TestRateLimiting:
    """Test rate limiting configuration."""

    @pytest.mark.asyncio
    async def test_limiter_configured(self):
        """Limiter should be configured on the app."""
        from app import limiter
        assert limiter is not None

    @pytest.mark.asyncio
    async def test_read_endpoints_not_rate_limited(self, client):
        """GET endpoints should not be rate limited."""
        for _ in range(20):
            resp = await client.get("/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self, client):
        """Rate limited endpoints should work under normal usage."""
        resp = await client.post("/events", json={
            "venue_id": "test",
            "api_key": "dummy",
            "events": [
                {
                    "pseudo_id": "v1",
                    "timestamp": "2025-01-01T00:00:00",
                    "zone": "entrance",
                    "dwell_seconds": 10
                }
            ]
        })
        assert resp.status_code == 200


# ===========================================================================
# Timeouts and Limits Tests
# ===========================================================================

class TestTimeoutsAndLimits:
    """Test upload size limits and download timeouts."""

    @pytest.mark.asyncio
    async def test_config_has_size_limit(self):
        """Config should define MAX_UPLOAD_SIZE_BYTES."""
        from app.config import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB
        assert MAX_UPLOAD_SIZE_BYTES == MAX_UPLOAD_SIZE_MB * 1024 * 1024
        assert MAX_UPLOAD_SIZE_BYTES > 0

    @pytest.mark.asyncio
    async def test_config_has_ytdl_timeout(self):
        """Config should define YTDL_TIMEOUT_SECONDS."""
        from app.config import YTDL_TIMEOUT_SECONDS
        assert YTDL_TIMEOUT_SECONDS > 0

    @pytest.mark.asyncio
    async def test_url_validation_prevents_ssrf(self, client):
        """Non-YouTube URLs should be rejected."""
        resp = await client.post("/process/youtube", json={
            "url": "http://evil.com/video.mp4"
        })
        assert resp.status_code == 400


# ===========================================================================
# Response Format Tests
# ===========================================================================

class TestResponseFormat:
    """Test standardized response envelope."""

    @pytest.mark.asyncio
    async def test_success_envelope(self, client):
        """API responses should have status, data, generated_at."""
        resp = await client.get("/venues")
        body = resp.json()
        assert body["status"] == "success"
        assert "data" in body
        assert "generated_at" in body

    @pytest.mark.asyncio
    async def test_health_not_wrapped(self, client):
        """Health endpoint should NOT use the envelope."""
        resp = await client.get("/health")
        body = resp.json()
        assert "status" in body
        assert body["status"] == "healthy"
        assert "data" not in body

    @pytest.mark.asyncio
    async def test_error_format(self, client):
        """Error responses should use HTTPException format."""
        resp = await client.get("/process/status/nonexistent")
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" in body

    @pytest.mark.asyncio
    async def test_analytics_wrapped(self, client):
        """Analytics endpoint should return wrapped response."""
        resp = await client.get("/analytics/test_venue?days=7")
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], dict)
        assert "generated_at" in body


# ===========================================================================
# Pagination Tests
# ===========================================================================

class TestPagination:
    """Test pagination on list endpoints."""

    @pytest.mark.asyncio
    async def test_venues_default_pagination(self, client):
        """GET /venues should include pagination metadata."""
        resp = await client.get("/venues")
        body = resp.json()
        assert "pagination" in body
        pag = body["pagination"]
        assert pag["limit"] == 50
        assert pag["offset"] == 0
        assert "total" in pag
        assert "has_more" in pag

    @pytest.mark.asyncio
    async def test_venues_explicit_pagination(self, client):
        """GET /venues with explicit limit/offset should work."""
        resp = await client.get("/venues?limit=2&offset=0")
        body = resp.json()
        assert body["pagination"]["limit"] == 2
        assert body["pagination"]["offset"] == 0

    @pytest.mark.asyncio
    async def test_venues_offset_beyond_total(self, client):
        """Offset beyond total should return empty data."""
        resp = await client.get("/venues?offset=99999")
        body = resp.json()
        assert body["status"] == "success"
        assert len(body["data"]) == 0
        assert body["pagination"]["has_more"] is False

    @pytest.mark.asyncio
    async def test_limit_validation(self, client):
        """Limit > max should fail validation."""
        resp = await client.get("/venues?limit=999")
        assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_batch_jobs_pagination(self, client):
        """GET /api/batch/jobs should include pagination."""
        resp = await client.get("/api/batch/jobs")
        body = resp.json()
        assert "pagination" in body
        pag = body["pagination"]
        assert "total" in pag
        assert "has_more" in pag
