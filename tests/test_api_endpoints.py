"""
API Endpoint Tests
==================
Tests all REST API endpoints for correct responses and error handling.
"""
import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_status_healthy(self, client):
        response = await client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestVenueEndpoints:
    """Test venue CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_venue_success(self, client):
        response = await client.post("/venues", json={
            "id": "new_venue",
            "name": "New Test Venue",
            "latitude": -26.0,
            "longitude": 28.0
        })
        assert response.status_code == 200
        data = response.json()
        assert data["venue_id"] == "new_venue"
        assert "api_key" in data
        assert len(data["api_key"]) == 64  # SHA256 hex

    @pytest.mark.asyncio
    async def test_create_venue_with_h3_zone(self, client):
        response = await client.post("/venues", json={
            "id": "geo_venue",
            "name": "Geo Venue",
            "latitude": -26.2041,
            "longitude": 28.0473
        })
        data = response.json()
        assert "h3_zone" in data
        assert data["h3_zone"] is not None

    @pytest.mark.asyncio
    async def test_create_duplicate_venue_fails(self, client, test_venue):
        response = await client.post("/venues", json={
            "id": "test_venue",
            "name": "Duplicate"
        })
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_list_venues(self, client, test_venue):
        response = await client.get("/venues")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_venue_without_location(self, client):
        response = await client.post("/venues", json={
            "id": "no_loc_venue",
            "name": "No Location Venue"
        })
        assert response.status_code == 200


class TestAnalyticsEndpoints:
    """Test analytics API endpoints."""

    @pytest.mark.asyncio
    async def test_analytics_empty_venue(self, client, test_venue):
        response = await client.get("/analytics/test_venue?days=7")
        assert response.status_code == 200
        data = response.json()
        assert data["unique_visitors"] == 0

    @pytest.mark.asyncio
    async def test_analytics_with_data(self, client, test_venue_with_data):
        response = await client.get("/analytics/test_venue?days=7")
        assert response.status_code == 200
        data = response.json()
        assert data["unique_visitors"] > 0

    @pytest.mark.asyncio
    async def test_analytics_nonexistent_venue(self, client):
        response = await client.get("/analytics/nonexistent?days=7")
        # Should return 200 with empty data, not 404
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_analytics_invalid_days_param(self, client, test_venue):
        response = await client.get("/analytics/test_venue?days=0")
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_analytics_days_too_large(self, client, test_venue):
        response = await client.get("/analytics/test_venue?days=100")
        assert response.status_code == 422


class TestSummaryEndpoint:
    """Test executive summary endpoint."""

    @pytest.mark.asyncio
    async def test_summary_returns_required_fields(self, client, test_venue):
        response = await client.get("/analytics/test_venue/summary?days=7")
        assert response.status_code == 200
        data = response.json()

        # Check required top-level fields
        assert "venue_id" in data
        assert "period" in data
        assert "current" in data
        assert "previous" in data
        assert "change" in data
        assert "insights" in data

    @pytest.mark.asyncio
    async def test_summary_current_has_all_metrics(self, client, test_venue):
        response = await client.get("/analytics/test_venue/summary?days=7")
        data = response.json()
        current = data["current"]

        required_fields = [
            "total_visitors", "unique_visitors", "return_visitors",
            "return_rate_percent", "avg_dwell_minutes", "peak_hour",
            "avg_engagement", "engaged_percent"
        ]
        for field in required_fields:
            assert field in current, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_summary_with_data(self, client, test_venue_with_data):
        response = await client.get("/analytics/test_venue/summary?days=7")
        data = response.json()
        assert data["current"]["unique_visitors"] > 0


class TestHourlyEndpoint:
    """Test hourly breakdown endpoint."""

    @pytest.mark.asyncio
    async def test_hourly_returns_24_hours(self, client, test_venue):
        response = await client.get("/analytics/test_venue/hourly")
        assert response.status_code == 200
        data = response.json()
        assert "hourly" in data
        assert len(data["hourly"]) == 24

    @pytest.mark.asyncio
    async def test_hourly_hours_are_sequential(self, client, test_venue):
        response = await client.get("/analytics/test_venue/hourly")
        data = response.json()
        hours = [h["hour"] for h in data["hourly"]]
        assert hours == list(range(24))


class TestDemographicsEndpoint:
    """Test demographics endpoint."""

    @pytest.mark.asyncio
    async def test_demographics_structure(self, client, test_venue):
        response = await client.get("/analytics/test_venue/demographics?days=7")
        assert response.status_code == 200
        data = response.json()
        assert "current" in data
        assert "gender" in data["current"]
        assert "age" in data["current"]

    @pytest.mark.asyncio
    async def test_demographics_with_data(self, client, test_venue_with_data):
        response = await client.get("/analytics/test_venue/demographics?days=7")
        data = response.json()
        # Should have gender data
        assert data["current"]["total"] > 0


class TestZonesEndpoint:
    """Test zone analytics endpoint."""

    @pytest.mark.asyncio
    async def test_zones_returns_list(self, client, test_venue):
        response = await client.get("/analytics/test_venue/zones?days=7")
        assert response.status_code == 200
        data = response.json()
        assert "zones" in data
        assert isinstance(data["zones"], list)

    @pytest.mark.asyncio
    async def test_zones_with_data(self, client, test_venue_with_data):
        response = await client.get("/analytics/test_venue/zones?days=7")
        data = response.json()
        assert len(data["zones"]) > 0


class TestBehaviorEndpoints:
    """Test behavior analytics endpoints."""

    @pytest.mark.asyncio
    async def test_behavior_structure(self, client, test_venue):
        response = await client.get("/analytics/test_venue/behavior?days=7")
        assert response.status_code == 200
        data = response.json()

        assert "engagement" in data
        assert "behavior_types" in data
        assert "postures" in data
        assert "body_orientation" in data

    @pytest.mark.asyncio
    async def test_behavior_hourly(self, client, test_venue):
        response = await client.get("/analytics/test_venue/behavior/hourly?days=7")
        assert response.status_code == 200
        data = response.json()
        assert "hourly_engagement" in data

    @pytest.mark.asyncio
    async def test_behavior_zones(self, client, test_venue):
        response = await client.get("/analytics/test_venue/behavior/zones?days=7")
        assert response.status_code == 200
        data = response.json()
        assert "zones" in data
        assert "insights" in data


class TestHeatmapEndpoint:
    """Test heatmap endpoint."""

    @pytest.mark.asyncio
    async def test_heatmap_structure(self, client, test_venue):
        response = await client.get("/analytics/test_venue/heatmap?weeks=1")
        assert response.status_code == 200
        data = response.json()

        assert "heatmap" in data
        assert "max_count" in data
        assert "peak_times" in data

    @pytest.mark.asyncio
    async def test_heatmap_has_all_days(self, client, test_venue):
        response = await client.get("/analytics/test_venue/heatmap?weeks=1")
        data = response.json()
        # API returns list of {day, hour, count, intensity} objects
        days_in_heatmap = set(item["day"] for item in data["heatmap"])
        expected_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        assert days_in_heatmap == expected_days


class TestExportEndpoint:
    """Test export endpoint."""

    @pytest.mark.asyncio
    async def test_export_json(self, client, test_venue):
        response = await client.get("/analytics/test_venue/export?format=json")
        assert response.status_code == 200
        data = response.json()
        assert "executive_summary" in data
        assert "demographics" in data
        assert "zone_performance" in data

    @pytest.mark.asyncio
    async def test_export_csv(self, client, test_venue):
        response = await client.get("/analytics/test_venue/export?format=csv")
        assert response.status_code == 200
        # Check content-type or content-disposition header
        content_type = response.headers.get("content-type", "")
        content_disp = response.headers.get("content-disposition", "")
        assert "csv" in content_type or "csv" in content_disp


class TestHTMLPages:
    """Test HTML page endpoints."""

    @pytest.mark.asyncio
    async def test_home_page(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_analytics_page(self, client):
        response = await client.get("/analytics")
        assert response.status_code == 200
        assert "Analytics" in response.text

    @pytest.mark.asyncio
    async def test_analytics_dashboard(self, client, test_venue):
        response = await client.get("/analytics-dashboard/test_venue")
        assert response.status_code == 200
        assert "Analytics Dashboard" in response.text

    @pytest.mark.asyncio
    async def test_process_page(self, client):
        response = await client.get("/process")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_uploads_page(self, client):
        response = await client.get("/uploads")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_map_page(self, client):
        response = await client.get("/map")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_report_page(self, client, test_venue):
        response = await client.get("/report/test_venue")
        assert response.status_code == 200
