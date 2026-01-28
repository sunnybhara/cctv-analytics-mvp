# Video Analytics MVP

Privacy-preserving venue analytics API. Receives pseudonymized visitor events from edge devices.

## Deploy to Railway

```bash
# 1. Install Railway CLI (if not already)
npm install -g @railway/cli

# 2. Login
railway login

# 3. Init project
railway init

# 4. Add PostgreSQL
railway add --database postgres

# 5. Deploy
railway up

# 6. Get your URL
railway domain
```

## Test Locally

```bash
# Install deps
pip install -r requirements.txt

# Run
python main.py

# In another terminal, run tests
python test_api.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard |
| GET | `/docs` | API documentation |
| POST | `/venues` | Create venue, returns API key |
| GET | `/venues` | List venues |
| POST | `/events` | Submit events (wrapped format) |
| POST | `/events/batch` | Submit events (array format) |
| GET | `/analytics/{venue_id}` | Get venue analytics |
| GET | `/analytics/{venue_id}/hourly` | Hourly breakdown |
| GET | `/health` | Health check |

## Submit Events (from edge device)

### Wrapped format (with API key):
```bash
curl -X POST https://your-app.railway.app/events \
  -H "Content-Type: application/json" \
  -d '{
    "venue_id": "blue_moon_bar",
    "api_key": "xxx",
    "events": [
      {
        "pseudo_id": "a3f8c2e1d9b4",
        "timestamp": "2026-01-28T21:30:00",
        "zone": "bar",
        "dwell_seconds": 180,
        "age_bracket": "30s",
        "gender": "M",
        "is_repeat": false
      }
    ]
  }'
```

### Array format:
```bash
curl -X POST https://your-app.railway.app/events/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "venue_id": "blue_moon_bar",
      "pseudo_id": "a3f8c2e1d9b4",
      "timestamp": "2026-01-28T21:30:00",
      "zone": "bar",
      "dwell_seconds": 180,
      "age_bracket": "30s",
      "gender": "M"
    }
  ]'
```

## Get Stats

```bash
curl https://your-app.railway.app/analytics/blue_moon_bar?days=7
```

## Architecture

```
EDGE (Venue)                    CLOUD (Railway)
+-------------+                +-------------------+
| CCTV        |                | FastAPI           |
|   |         |                |   |               |
| Detection   |  ---events---> | PostgreSQL        |
|   |         |                |   |               |
| Pseudonym   |                | Dashboard         |
+-------------+                +-------------------+
```

Heavy processing on edge. Cloud only receives lightweight JSON events.
