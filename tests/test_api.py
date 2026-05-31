#!/usr/bin/env python3
"""
Test script for Video Analytics MVP API
Run locally first, then update BASE_URL for Railway deployment.
"""

import httpx
import asyncio
from datetime import datetime, timedelta
import random

# Change to Railway URL after deploy
BASE_URL = "http://localhost:8000"


async def test_health():
    """Test health endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{BASE_URL}/health")
            print(f"Health: {r.status_code} - {r.json()}")
            return r.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False


async def generate_mock_events(venue_id: str, count: int = 100):
    """Generate realistic mock events."""
    events = []
    zones = ["entrance", "bar", "seating", "restroom", "exit"]
    age_brackets = ["20s", "30s", "40s", "50+"]
    genders = ["M", "F"]

    # Generate visitor pool (some will repeat)
    visitor_pool = [f"visitor_{random.randint(1000, 9999)}" for _ in range(count // 3)]

    for i in range(count):
        # 30% chance of repeat visitor
        if random.random() < 0.3 and visitor_pool:
            pseudo_id = random.choice(visitor_pool)
            is_repeat = True
        else:
            pseudo_id = f"visitor_{random.randint(10000, 99999)}"
            visitor_pool.append(pseudo_id)
            is_repeat = False

        events.append({
            "venue_id": venue_id,
            "pseudo_id": pseudo_id,
            "timestamp": (datetime.now() - timedelta(
                hours=random.randint(0, 168),  # Last week
                minutes=random.randint(0, 59)
            )).isoformat(),
            "zone": random.choice(zones),
            "dwell_seconds": random.randint(30, 3600),
            "age_bracket": random.choice(age_brackets),
            "gender": random.choice(genders),
            "is_repeat": is_repeat
        })
    return events


async def test_batch_ingest(venue_id: str, count: int = 100):
    """Test batch event ingestion."""
    events = await generate_mock_events(venue_id, count)

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"{BASE_URL}/events/batch",
                json=events,
                timeout=30.0
            )
            print(f"Batch ingest ({count} events): {r.status_code} - {r.json()}")
            return r.status_code == 200
        except Exception as e:
            print(f"Batch ingest failed: {e}")
            return False


async def test_wrapped_batch_ingest(venue_id: str, count: int = 50):
    """Test wrapped batch format (with api_key)."""
    events = await generate_mock_events(venue_id, count)

    # Convert to wrapped format
    event_list = []
    for e in events:
        event_list.append({
            "pseudo_id": e["pseudo_id"],
            "timestamp": e["timestamp"],
            "zone": e["zone"],
            "dwell_seconds": e["dwell_seconds"],
            "age_bracket": e["age_bracket"],
            "gender": e["gender"],
            "is_repeat": e.get("is_repeat", False)
        })

    payload = {
        "venue_id": venue_id,
        "api_key": "test_key",
        "events": event_list
    }

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"{BASE_URL}/events",
                json=payload,
                timeout=30.0
            )
            print(f"Wrapped batch ingest ({count} events): {r.status_code} - {r.json()}")
            return r.status_code == 200
        except Exception as e:
            print(f"Wrapped batch ingest failed: {e}")
            return False


async def test_analytics(venue_id: str):
    """Test analytics endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{BASE_URL}/analytics/{venue_id}?days=7")
            print(f"Analytics: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                print(f"  Total visitors: {data['total_visitors']}")
                print(f"  Unique visitors: {data['unique_visitors']}")
                print(f"  Repeat rate: {data['repeat_rate']}%")
                print(f"  Avg dwell: {data['avg_dwell_minutes']} min")
                print(f"  Peak hour: {data['peak_hour']}")
                print(f"  Gender split: {data['gender_split']}")
                print(f"  Age distribution: {data['age_distribution']}")
            return r.status_code == 200
        except Exception as e:
            print(f"Analytics failed: {e}")
            return False


async def test_hourly(venue_id: str):
    """Test hourly breakdown endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            r = await client.get(f"{BASE_URL}/analytics/{venue_id}/hourly?date={today}")
            print(f"Hourly breakdown: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                # Show hours with visitors
                active_hours = [h for h in data["hourly"] if h["visitors"] > 0]
                print(f"  Active hours: {len(active_hours)}")
                for h in active_hours[:5]:
                    print(f"    {h['hour']:02d}:00 - {h['visitors']} visitors ({h['unique']} unique)")
            return r.status_code == 200
        except Exception as e:
            print(f"Hourly breakdown failed: {e}")
            return False


async def test_venues():
    """Test venues endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{BASE_URL}/venues")
            print(f"List venues: {r.status_code} - {len(r.json())} venues")
            return r.status_code == 200
        except Exception as e:
            print(f"List venues failed: {e}")
            return False


async def main():
    venue_id = "blue_moon_bar"

    print("=" * 50)
    print("CCTV Analytics API Test")
    print("=" * 50)
    print(f"Base URL: {BASE_URL}")
    print(f"Venue: {venue_id}")
    print("=" * 50)

    results = {}

    print("\n1. Testing health endpoint...")
    results["health"] = await test_health()

    print("\n2. Testing venues endpoint...")
    results["venues"] = await test_venues()

    print("\n3. Ingesting 100 events (batch format)...")
    results["batch"] = await test_batch_ingest(venue_id, 100)

    print("\n4. Ingesting 50 events (wrapped format)...")
    results["wrapped"] = await test_wrapped_batch_ingest(venue_id, 50)

    print("\n5. Querying analytics...")
    results["analytics"] = await test_analytics(venue_id)

    print("\n6. Querying hourly breakdown...")
    results["hourly"] = await test_hourly(venue_id)

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 50)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
