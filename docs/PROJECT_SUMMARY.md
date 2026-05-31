# CCTV Analytics MVP - Complete Project Summary

## What We Built

A **real-time retail analytics platform** that processes CCTV/video footage using state-of-the-art ML models to detect, track, and analyze visitors — delivering demographics, behavior patterns, re-identification, and engagement scoring through a full-stack web application with 45 API endpoints and 8 interactive dashboards.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                         │
│                         (main.py → app/)                           │
├────────────┬────────────┬────────────┬─────────────────────────────┤
│   13 API   │  8 HTML    │  Video     │  ML Pipeline                │
│   Routers  │  Dashboards│  Processing│  (YOLO11 + InsightFace +    │
│   (JSON)   │  (Chart.js)│  Queue     │   YOLO11-Pose)              │
├────────────┴────────────┴────────────┴─────────────────────────────┤
│                    SQLite / PostgreSQL                              │
│              (7 tables, auto-created via SQLAlchemy)               │
└─────────────────────────────────────────────────────────────────────┘
```

### ML Pipeline Flow

```
Video Input (YouTube URL / File Upload)
    │
    ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  YOLO11n          │     │  InsightFace     │     │  InsightFace     │
│  Person Detection │────▶│  buffalo_l       │────▶│  buffalo_l       │
│  + BoT-SORT       │     │  SCRFD Face Det  │     │  Age + Gender    │
│  Multi-Object     │     │  ArcFace 512-dim │     │  Single-Pass     │
│  Tracking         │     │  Embeddings      │     │  Demographics    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
    │                                                      │
    ▼                                                      ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  YOLO11-Pose     │     │  Engagement      │     │  Visitor         │
│  17 COCO         │────▶│  Scoring         │────▶│  Re-ID           │
│  Keypoints       │     │  (0-100 scale)   │     │  Matching        │
│  Multi-Person    │     │  + Behavior Type │     │  (cosine sim)    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────────┐
                                                  │  SQLite/Postgres │
                                                  │  Events + ReID   │
                                                  │  Embeddings      │
                                                  └──────────────────┘
```

---

## ML Models

| Component | Model | What It Does | Key Metric |
|-----------|-------|-------------|------------|
| Person Detection | **YOLO11n** (6MB) | Detects people in frames | mAP 39.5% on COCO |
| Multi-Object Tracking | **BoT-SORT** (ultralytics built-in) | Persistent track IDs across frames | HOTA 65.0 on MOT17 |
| Face Detection | **SCRFD** (InsightFace buffalo_l) | Locates faces within person crops | 90%+ on WIDER FACE Hard |
| Face Embeddings | **ArcFace** (InsightFace buffalo_l) | 512-dim face vectors for re-ID | 96% TAR on IJB-C |
| Age + Gender | **InsightFace buffalo_l** | Single-pass demographics | Continuous age + binary gender |
| Pose Estimation | **YOLO11n-Pose** (6MB) | 17 COCO keypoints per person | Native multi-person |

---

## Project Structure

```
cctv-analytics-mvp/                    # 8,395 lines of Python
├── main.py                            # Thin entry point (25 lines)
├── reid.py                            # Face embeddings + visitor matching (295 lines)
├── demographics.py                    # Age/gender estimation (396 lines)
├── behavior.py                        # Pose analysis + engagement scoring (517 lines)
├── botsort.yaml                       # BoT-SORT tracker config
├── requirements.txt                   # 18 dependencies
│
├── app/                               # FastAPI application package
│   ├── __init__.py                    # App factory, lifespan, CORS, router registration
│   ├── config.py                      # DATABASE_URL, CORS_ORIGINS, ALLOWED_VIDEO_DOMAINS
│   ├── database.py                    # 7 SQLAlchemy table definitions
│   ├── schemas.py                     # 5 Pydantic request/response models
│   ├── state.py                       # Shared state (processing_jobs dict, queue lock)
│   │
│   ├── routers/                       # 13 API router modules
│   │   ├── health.py                  # GET /health
│   │   ├── venues.py                  # Venue CRUD
│   │   ├── events.py                  # Event ingestion
│   │   ├── analytics.py               # Core analytics + hourly
│   │   ├── advanced_analytics.py      # Demographics, zones, trends, heatmap, export
│   │   ├── behavior.py               # Engagement, behavior types, pose analysis
│   │   ├── alerts.py                  # Anomaly detection + alert management
│   │   ├── benchmarks.py             # Venue comparison + industry benchmarks
│   │   ├── visitors.py               # ReID visitor profiles + loyalty
│   │   ├── video_processing.py       # YouTube/upload processing + status
│   │   ├── batch.py                  # Multi-video queue processing
│   │   ├── map_api.py                # Geo venue data for map
│   │   └── pages.py                  # 7 HTML dashboard pages (3,316 lines)
│   │
│   └── video/                         # Video processing subsystem
│       ├── deps.py                    # Lazy loading of cv2, YOLO, yt-dlp
│       ├── download.py                # YouTube video downloader
│       ├── models.py                  # Model cache + preload_models() warmup
│       ├── pipeline.py                # Main processing loop (523 lines)
│       ├── embeddings.py              # Sync DB ops for visitor embeddings
│       ├── helpers.py                 # Pseudo-ID, zones, demographics wrapper
│       └── queue.py                   # Background batch queue worker
│
├── tests/                             # 85 tests across 4 files
│   ├── conftest.py                    # Test fixtures, async DB setup
│   ├── test_api_endpoints.py          # 35 tests, 11 classes
│   ├── test_behavior.py              # 14 tests, 3 classes
│   ├── test_edge_cases.py            # 22 tests, 7 classes
│   └── test_reid.py                  # 14 tests, 3 classes
│
└── main_original.py                   # Pre-refactor backup (6,943-line monolith)
```

---

## API Endpoints (45 total)

### Venue Management
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/venues` | Create venue with optional geo-location |
| `GET` | `/venues` | List all venues |

### Event Ingestion
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/events` | Receive batch events from edge device |
| `POST` | `/events/batch` | Receive events as simple array |

### Analytics
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/analytics/{venue_id}` | Full analytics with confidence scoring |
| `GET` | `/analytics/{venue_id}/hourly` | Hourly breakdown for a date |
| `GET` | `/analytics/{venue_id}/demographics` | Nielsen-style demographic breakdown |
| `GET` | `/analytics/{venue_id}/zones` | Zone performance metrics |
| `GET` | `/analytics/{venue_id}/trends` | Weekly trends over configurable weeks |
| `GET` | `/analytics/{venue_id}/summary` | Executive summary with insights |
| `GET` | `/analytics/{venue_id}/heatmap` | Hourly heatmap by day-of-week |
| `GET` | `/analytics/{venue_id}/export` | Export as JSON or CSV |

### Behavior Analytics
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/analytics/{venue_id}/behavior` | Engagement scores + behavior types |
| `GET` | `/analytics/{venue_id}/behavior/hourly` | Hourly engagement patterns |
| `GET` | `/analytics/{venue_id}/behavior/zones` | Zone-level engagement |

### Visitor Re-Identification
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/visitors/{venue_id}` | Known visitors with loyalty scores |
| `GET` | `/api/visitors/{venue_id}/stats` | Loyalty distribution stats |
| `GET` | `/api/visitors/{venue_id}/history/{visitor_id}` | Individual visit history |
| `GET` | `/api/visitors/{venue_id}/returning` | Return visitor analytics |

### Alerts & Anomaly Detection
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/alerts` | List alerts (filterable) |
| `POST` | `/api/alerts/{alert_id}/acknowledge` | Acknowledge an alert |
| `POST` | `/api/alerts/check/{venue_id}` | Trigger anomaly detection |

### Video Processing
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/process/youtube` | Process YouTube video |
| `POST` | `/process/upload` | Process uploaded video file |
| `GET` | `/process/status/{job_id}` | Job status polling |

### Batch Processing
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/batch/upload` | Upload multiple videos |
| `POST` | `/api/batch/url` | Queue multiple YouTube URLs |
| `GET` | `/api/batch/jobs` | List all jobs |
| `GET` | `/api/batch/jobs/{job_id}` | Job details with live progress |
| `DELETE` | `/api/batch/jobs/{job_id}` | Cancel pending job |
| `GET` | `/api/batch/stats` | Queue statistics |

### Benchmarks & Map
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/benchmark/venues` | Compare venues side by side |
| `GET` | `/api/benchmark/industry` | Industry benchmarks by type |
| `GET` | `/api/map/venues` | All venues with counts for map |
| `GET` | `/health` | Health check |

### HTML Dashboards (8 pages)
| Path | Description |
|------|-------------|
| `/` | Home dashboard with live stats |
| `/analytics` | Analytics home listing all venues |
| `/process` | Video upload / YouTube URL form |
| `/dashboard/{venue_id}` | Basic venue dashboard |
| `/analytics-dashboard/{venue_id}` | Full analytics with Chart.js graphs |
| `/uploads` | Batch upload queue manager |
| `/map` | Interactive Leaflet.js venue map |
| `/report/{venue_id}` | Printable behavior analysis report |

---

## Database Schema (7 tables)

### `events` — Core visitor event data
| Column | Type | Description |
|--------|------|-------------|
| id | Integer PK | Auto-increment |
| venue_id | String | Which venue |
| pseudo_id | String | Anonymized track ID |
| timestamp | DateTime | When detected |
| zone | String | Store zone (front/middle/back/left/right) |
| dwell_seconds | Float | Time spent |
| age_bracket | String | 20-29, 30-39, 40-49, etc. |
| gender | String | M/F |
| is_repeat | Boolean | Return visitor flag |
| track_frames | Integer | Frames tracked |
| detection_conf | Float | YOLO confidence |
| engagement_score | Float | 0-100 engagement |
| behavior_type | String | engaged/browsing/waiting/passing |
| body_orientation | Float | -1 to 1 (away to facing) |
| posture | String | upright/leaning/arms_crossed |

### `venues` — Registered locations
| Column | Type | Description |
|--------|------|-------------|
| id | String PK | Venue identifier |
| name | String | Display name |
| api_key | String | Auth key (generated, not yet enforced) |
| latitude/longitude | Float | Geo coordinates |
| h3_zone | String | H3 hex index |
| city/country | String | Location |
| venue_type | String | retail/restaurant/etc. |

### `visitor_embeddings` — Face embeddings for re-ID
| Column | Type | Description |
|--------|------|-------------|
| visitor_id | String | Unique face hash |
| embedding | LargeBinary | 512-dim ArcFace vector |
| embedding_model | String | "arcface" |
| visit_count | Integer | Times seen |
| age_bracket/gender | String | Demographics |

### `jobs` — Batch processing queue
### `alerts` — Anomaly detection alerts
### `daily_stats` — Aggregated daily stats (unused)
### `visitor_sessions` — Session tracking (unused)

---

## Features

### Core Analytics
- **Visitor counting** with confidence intervals and data quality scoring
- **Demographics**: Age bracket distribution + gender split (InsightFace single-pass)
- **Dwell time**: Track-based measurement (not fabricated)
- **Zone analytics**: Front/middle/back/left/right performance
- **Hourly breakdown**: Traffic patterns by hour
- **Weekly trends**: Growth rates over configurable periods
- **Heatmap**: Day-of-week x hour-of-day traffic matrix
- **Executive summary**: Auto-generated insights with actionable recommendations
- **Export**: JSON and CSV download

### Behavior & Engagement
- **Engagement scoring** (0-100): Body orientation (35%) + posture (25%) + stationarity (25%) + pose confidence (15%)
- **Behavior classification**: Engaged / Browsing / Waiting / Passing
- **Posture detection**: Upright / Leaning / Arms crossed / Hands on hips
- **Body orientation**: Facing camera vs sideways vs away
- **Hourly engagement patterns**: When are visitors most engaged?
- **Zone engagement**: Which areas drive highest engagement?

### Visitor Re-Identification
- **Face embedding extraction**: 512-dim ArcFace vectors per visitor
- **Cosine similarity matching**: Threshold 0.68 for same-person detection
- **Return visitor tracking**: Visit count, frequency, loyalty tier
- **Loyalty scoring**: New / Occasional / Regular / Loyal / VIP
- **Visit history**: Per-visitor timeline across sessions

### Video Processing
- **YouTube URL processing**: Paste URL, auto-download + analyze
- **File upload**: Direct video file upload
- **Batch processing**: Queue multiple videos with priority
- **Live progress**: Real-time frame-by-frame status
- **Background processing**: Non-blocking via thread pool

### Alerts & Anomaly Detection
- **Traffic spike detection**: Unusual visitor volume alerts
- **Traffic drop detection**: Below-average periods flagged
- **Unusual hour alerts**: Activity outside normal hours
- **Alert management**: Acknowledge, filter by severity/venue

### Multi-Venue
- **Venue benchmarking**: Side-by-side comparison
- **Industry benchmarks**: Compare against venue type averages
- **Interactive map**: Leaflet.js with venue markers + visitor counts
- **H3 geo-indexing**: Hex-based spatial indexing

### Dashboards (8 interactive HTML pages)
- **Home**: Live stats, recent activity, quick actions
- **Analytics Dashboard**: Chart.js graphs (demographics, zones, trends, heatmap)
- **Behavior Report**: Printable Nielsen-style engagement report
- **Video Processing**: Upload form with live progress bar
- **Batch Manager**: Queue management with job list
- **Venue Map**: Interactive Leaflet.js map
- **Analytics Home**: Venue selector
- **Basic Dashboard**: Simple metrics view

---

## Security Measures Applied

| Fix | Description |
|-----|-------------|
| XSS Prevention | `html.escape()` on all venue_id HTML interpolation |
| CORS | Environment-configurable origins, credentials disabled |
| SSRF Prevention | URL domain whitelist on YouTube/batch URL endpoints |
| Path Traversal | `os.path.basename()` on uploaded filenames |
| SQL Injection | All queries parameterized via SQLAlchemy |
| Error Leak | Generic error messages, tracebacks logged not exposed |
| Data Integrity | Deterministic dwell times (no random fabrication) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI 0.128, Python 3.13, Uvicorn |
| **Database** | SQLite (dev) / PostgreSQL (prod) via databases + SQLAlchemy |
| **Detection** | Ultralytics YOLO11n + BoT-SORT |
| **Face AI** | InsightFace buffalo_l (SCRFD + ArcFace) |
| **Pose** | YOLO11n-Pose (17 COCO keypoints) |
| **Video** | OpenCV + yt-dlp |
| **Frontend** | Inline HTML + Chart.js + Leaflet.js |
| **Testing** | pytest + pytest-asyncio (85 tests) |
| **Deployment** | Railway / Render ready (railway.toml, render.yaml) |

---

## Running Locally

```bash
# Clone and setup
git clone https://github.com/sunnybhara/cctv-analytics-mvp.git
cd cctv-analytics-mvp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start server (models auto-download on first use)
uvicorn main:app --host 0.0.0.0 --port 8000

# Open in browser
# Home:       http://localhost:8000/
# API Docs:   http://localhost:8000/docs
# Process:    http://localhost:8000/process
```

### Quick Test

```bash
# Create a venue
curl -X POST http://localhost:8000/venues \
  -H "Content-Type: application/json" \
  -d '{"id":"my_store","name":"My Store"}'

# Process a YouTube video
curl -X POST http://localhost:8000/process/youtube \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID","venue_id":"my_store"}'

# Check status
curl http://localhost:8000/process/status/JOB_ID

# View analytics
curl http://localhost:8000/analytics/my_store
```

---

## Test Results

```
85 passed, 0 failures
41/41 files syntax validated
18/18 ML pipeline integration checks passing
10/10 cross-module imports verified
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Total Python code | 8,395 lines |
| API endpoints | 45 (38 JSON + 7 HTML) |
| Database tables | 7 |
| ML models | 3 (YOLO11n, InsightFace buffalo_l, YOLO11n-Pose) |
| Test coverage | 85 tests across 24 test classes |
| Router modules | 13 |
| Pydantic models | 5 |
| Dependencies | 18 packages |
