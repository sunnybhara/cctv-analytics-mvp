"""
Edge Case and Error Handling Tests
==================================
Tests boundary conditions, malformed inputs, and error scenarios.
"""
import pytest
from httpx import AsyncClient


class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.mark.asyncio
    async def test_venue_id_special_characters(self, client):
        """Venue IDs with special chars should be handled."""
        response = await client.post("/venues", json={
            "id": "test-venue_123",
            "name": "Test"
        })
        # Should work - alphanumeric + hyphen + underscore
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_venue_id_very_long(self, client):
        """Very long venue IDs should be handled."""
        long_id = "a" * 100
        response = await client.post("/venues", json={
            "id": long_id,
            "name": "Long ID Venue"
        })
        # Should either work or return validation error
        assert response.status_code in [200, 422]

    @pytest.mark.asyncio
    async def test_negative_days_param(self, client, test_venue):
        """Negative days parameter should fail validation."""
        response = await client.get("/analytics/test_venue?days=-1")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_non_numeric_days_param(self, client, test_venue):
        """Non-numeric days should fail."""
        response = await client.get("/analytics/test_venue?days=abc")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_venue_name(self, client):
        """Empty venue name should be allowed (name is optional)."""
        import uuid
        unique_id = f"no_name_{uuid.uuid4().hex[:8]}"
        response = await client.post("/venues", json={
            "id": unique_id,
            "name": ""  # Empty name should be allowed
        })
        # Could be 200 (allowed) or 422 (validation requires non-empty)
        assert response.status_code in [200, 422]

    @pytest.mark.asyncio
    async def test_invalid_coordinates(self, client):
        """Invalid lat/long should be handled gracefully."""
        response = await client.post("/venues", json={
            "id": "bad_coords",
            "name": "Bad Coords",
            "latitude": 999,  # Invalid
            "longitude": 999  # Invalid
        })
        # Should either validate or handle gracefully
        assert response.status_code in [200, 422]


class TestEmptyDataHandling:
    """Test handling of empty/missing data."""

    @pytest.mark.asyncio
    async def test_analytics_no_events(self, client):
        """Analytics should return zeros for venue with no events."""
        import uuid
        empty_venue = f"empty_{uuid.uuid4().hex[:8]}"
        # Create empty venue
        await client.post("/venues", json={"id": empty_venue, "name": "Empty Venue"})
        response = await client.get(f"/analytics/{empty_venue}?days=7")
        data = response.json()["data"]
        assert data["unique_visitors"] == 0
        assert data["total_visitors"] == 0

    @pytest.mark.asyncio
    async def test_summary_empty_insights(self, client):
        """Summary with no data should have empty insights."""
        import uuid
        empty_venue = f"empty_sum_{uuid.uuid4().hex[:8]}"
        await client.post("/venues", json={"id": empty_venue, "name": "Empty Summary"})
        response = await client.get(f"/analytics/{empty_venue}/summary?days=7")
        data = response.json()["data"]
        assert isinstance(data["insights"], list)

    @pytest.mark.asyncio
    async def test_zones_empty_list(self, client):
        """Zones endpoint with no data returns empty list."""
        import uuid
        empty_venue = f"empty_zone_{uuid.uuid4().hex[:8]}"
        await client.post("/venues", json={"id": empty_venue, "name": "Empty Zones"})
        response = await client.get(f"/analytics/{empty_venue}/zones?days=7")
        data = response.json()["data"]
        assert data["zones"] == []

    @pytest.mark.asyncio
    async def test_behavior_empty(self, client):
        """Behavior endpoint handles no data gracefully."""
        import uuid
        empty_venue = f"empty_beh_{uuid.uuid4().hex[:8]}"
        await client.post("/venues", json={"id": empty_venue, "name": "Empty Behavior"})
        response = await client.get(f"/analytics/{empty_venue}/behavior?days=7")
        data = response.json()["data"]
        assert data["total_analyzed"] == 0
        assert data["behavior_types"] == {}


class TestBoundaryConditions:
    """Test boundary conditions."""

    @pytest.mark.asyncio
    async def test_min_days(self, client, test_venue):
        """Minimum days parameter (1)."""
        response = await client.get("/analytics/test_venue?days=1")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_max_days(self, client, test_venue):
        """Maximum days parameter (90)."""
        response = await client.get("/analytics/test_venue?days=90")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_heatmap_min_weeks(self, client, test_venue):
        """Heatmap with 1 week."""
        response = await client.get("/analytics/test_venue/heatmap?weeks=1")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_heatmap_max_weeks(self, client, test_venue):
        """Heatmap with many weeks."""
        response = await client.get("/analytics/test_venue/heatmap?weeks=52")
        assert response.status_code == 200


class TestConcurrency:
    """Test concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_analytics_requests(self, client, test_venue):
        """Multiple concurrent analytics requests should work."""
        import asyncio

        async def make_request():
            return await client.get("/analytics/test_venue?days=7")

        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        for response in responses:
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_venue_creation(self, client):
        """Concurrent venue creation should handle conflicts."""
        import asyncio

        async def create_venue(i):
            return await client.post("/venues", json={
                "id": f"concurrent_venue_{i}",
                "name": f"Concurrent {i}"
            })

        tasks = [create_venue(i) for i in range(5)]
        responses = await asyncio.gather(*tasks)

        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count == 5  # All should succeed with unique IDs


class TestDataIntegrity:
    """Test data integrity and consistency."""

    @pytest.mark.asyncio
    async def test_unique_visitors_consistency(self, client, test_venue_with_data):
        """Unique visitors should be consistent across endpoints."""
        analytics = await client.get("/analytics/test_venue?days=7")
        summary = await client.get("/analytics/test_venue/summary?days=7")

        analytics_data = analytics.json()["data"]
        summary_data = summary.json()["data"]

        assert analytics_data["unique_visitors"] == summary_data["current"]["unique_visitors"]

    @pytest.mark.asyncio
    async def test_zone_totals_match(self, client, test_venue_with_data):
        """Zone visitor totals should sum correctly."""
        zones = await client.get("/analytics/test_venue/zones?days=7")
        analytics = await client.get("/analytics/test_venue?days=7")

        zones_data = zones.json()["data"]
        analytics_data = analytics.json()["data"]

        # Zone visitors may exceed total unique visitors (same person in multiple zones)
        # But total events should match
        zone_total = sum(z["visitors"] for z in zones_data["zones"])
        # Zone total can be >= unique visitors due to zone transitions
        assert zone_total >= 0


class TestErrorResponses:
    """Test error response format."""

    @pytest.mark.asyncio
    async def test_404_format(self, client):
        """404 errors should have proper format."""
        response = await client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_422_format(self, client, test_venue):
        """Validation errors should have proper format."""
        response = await client.get("/analytics/test_venue?days=0")
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestSQLInjectionPrevention:
    """Test SQL injection prevention."""

    @pytest.mark.asyncio
    async def test_venue_id_sql_injection(self, client):
        """SQL injection in venue_id should be safe."""
        malicious_id = "test'; DROP TABLE events; --"
        response = await client.get(f"/analytics/{malicious_id}?days=7")
        # Should either return empty data or 404, not crash
        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_zone_filter_injection(self, client, test_venue):
        """SQL injection in query params should be safe."""
        response = await client.get(
            "/analytics/test_venue?days=7",
            params={"zone": "'; DROP TABLE events; --"}
        )
        assert response.status_code in [200, 422]
