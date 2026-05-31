# CCTV Analytics - Full Product Vision

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VIDEO INGESTION                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Single Upload │ Batch Upload │ Live Stream │ API Push │ Edge Device   │
│       ↓              ↓              ↓            ↓            ↓         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    METADATA EXTRACTION                           │   │
│  │  • GPS from EXIF/video metadata                                  │   │
│  │  • Timestamp normalization                                       │   │
│  │  • Camera ID / Venue ID mapping                                  │   │
│  │  • Manual lat/long entry fallback                                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         ZONE ASSIGNMENT                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Lat/Long → H3 Hexagon Grid (like Uber)                                 │
│                                                                          │
│  Zone Hierarchy:                                                         │
│  ├── Country (South Africa, Nigeria, Kenya...)                          │
│  ├── City (Johannesburg, Lagos, Nairobi...)                             │
│  ├── District (CBD, Township, Suburb...)                                │
│  ├── Micro-Zone (H3 resolution 9 = ~150m hexagon)                       │
│  └── Venue (specific location)                                          │
│                                                                          │
│  Zone Enrichment (external data):                                       │
│  • Affluence score (from census/economic data)                          │
│  • Population density                                                    │
│  • Business type concentration                                           │
│  • Safety index                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      INTELLIGENT MODEL ROUTER                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input Analysis:                                                         │
│  ├── Video quality (resolution, lighting, FPS)                          │
│  ├── Crowd density (sparse < 10, medium 10-50, dense > 50)              │
│  ├── Scene type (indoor bar, outdoor market, retail store)              │
│  └── Required analytics (count only, demographics, behavior)            │
│                                                                          │
│  Model Selection:                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ IF crowd_density > 50:                                           │    │
│  │     USE crowd_counting_model (CSRNet)                            │    │
│  │ ELIF crowd_density > 10:                                         │    │
│  │     USE yolov8m (medium) + DeepSORT                              │    │
│  │ ELSE:                                                            │    │
│  │     USE yolov8n (nano) + DeepSORT                                │    │
│  │                                                                   │    │
│  │ IF demographics_required AND face_visible:                        │    │
│  │     USE insightface + fairface                                    │    │
│  │                                                                   │    │
│  │ IF behavior_required:                                             │    │
│  │     USE mediapipe_pose + action_classifier                        │    │
│  │                                                                   │    │
│  │ IF scene_context_required:                                        │    │
│  │     USE CLIP for zero-shot classification                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       PROCESSING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Layer 1: DETECTION                                                      │
│  ├── YOLOv8 (person detection)                                          │
│  ├── RetinaFace (face detection)                                        │
│  └── CSRNet (dense crowd estimation)                                    │
│                                                                          │
│  Layer 2: TRACKING                                                       │
│  ├── DeepSORT (multi-object tracking)                                   │
│  ├── ByteTrack (fast tracking alternative)                              │
│  └── StrongSORT (high accuracy option)                                  │
│                                                                          │
│  Layer 3: IDENTIFICATION                                                 │
│  ├── OSNet (person re-identification)                                   │
│  ├── ArcFace (face re-identification)                                   │
│  └── DINO (self-supervised features)                                    │
│                                                                          │
│  Layer 4: ATTRIBUTES                                                     │
│  ├── FairFace (age, gender, ethnicity)                                  │
│  ├── ViT-Age (age classification)                                       │
│  ├── DeepFashion (clothing attributes)                                  │
│  └── Emotion (facial expression - limited)                              │
│                                                                          │
│  Layer 5: BEHAVIOR                                                       │
│  ├── MediaPipe Pose (body keypoints)                                    │
│  ├── SlowFast (action recognition)                                      │
│  ├── Gaze estimation (attention direction)                              │
│  └── Social distance (group detection)                                  │
│                                                                          │
│  Layer 6: SCENE CONTEXT                                                  │
│  ├── CLIP (scene classification)                                        │
│  ├── Places365 (venue type)                                             │
│  └── Object detection (drinks, phones, bags)                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       DATA AGGREGATION                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Per-Video Metrics:                                                      │
│  • Unique visitors, peak concurrent, dwell times                        │
│  • Demographics breakdown                                                │
│  • Behavior patterns                                                     │
│  • Confidence scores                                                     │
│                                                                          │
│  Per-Venue Aggregation:                                                  │
│  • Daily/weekly/monthly trends                                           │
│  • Hour-by-hour patterns                                                 │
│  • Visitor profiles                                                      │
│                                                                          │
│  Per-Zone Aggregation:                                                   │
│  • Zone traffic comparison                                               │
│  • Cross-venue patterns                                                  │
│  • Demographic distribution by area                                      │
│                                                                          │
│  City/Country Level:                                                     │
│  • Market insights                                                       │
│  • Industry benchmarks                                                   │
│  • Economic indicators                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       MAP VISUALIZATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                      INTERACTIVE MAP                           │     │
│  │  ┌─────────────────────────────────────────────────────────┐   │     │
│  │  │                                                         │   │     │
│  │  │    [Filters: Time | Demographics | Behavior | Zone]     │   │     │
│  │  │                                                         │   │     │
│  │  │         🔴 High Traffic    🟡 Medium    🟢 Low          │   │     │
│  │  │                                                         │   │     │
│  │  │    ┌──────────────────────────────────────────────┐    │   │     │
│  │  │    │                                              │    │   │     │
│  │  │    │           [MAP OF AFRICA]                    │    │   │     │
│  │  │    │                                              │    │   │     │
│  │  │    │    🔴 Lagos CBD         🟡 Nairobi Central   │    │   │     │
│  │  │    │         ↳ 15k/day           ↳ 8k/day         │    │   │     │
│  │  │    │                                              │    │   │     │
│  │  │    │    🟢 Johannesburg South  🔴 Cape Town V&A   │    │   │     │
│  │  │    │         ↳ 2k/day              ↳ 20k/day      │    │   │     │
│  │  │    │                                              │    │   │     │
│  │  │    └──────────────────────────────────────────────┘    │   │     │
│  │  │                                                         │   │     │
│  │  │    Click zone to drill down → Venues → Cameras          │   │     │
│  │  │                                                         │   │     │
│  │  └─────────────────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Drill-Down Levels:                                                      │
│  1. Country → See all cities                                             │
│  2. City → See all zones (H3 hexagons)                                   │
│  3. Zone → See all venues                                                │
│  4. Venue → See all cameras                                              │
│  5. Camera → See analytics dashboard                                     │
│                                                                          │
│  Map Layers (toggleable):                                                │
│  □ Traffic heatmap                                                       │
│  □ Demographics overlay                                                  │
│  □ Affluence index                                                       │
│  □ Time-of-day patterns                                                  │
│  □ Competitor density                                                    │
│  □ Growth trends                                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

---

## Hugging Face Models Registry

### Detection & Counting
| Model | HuggingFace ID | Use Case | When to Use |
|-------|----------------|----------|-------------|
| YOLOv8-nano | `ultralytics/yolov8n` | Fast person detection | < 10 people, edge devices |
| YOLOv8-medium | `ultralytics/yolov8m` | Balanced detection | 10-50 people |
| YOLOv8-large | `ultralytics/yolov8l` | High accuracy | Critical venues |
| DETR | `facebook/detr-resnet-50` | Transformer detection | Complex scenes |
| CSRNet | `geekyrakshit/csrnet-crowd-counting` | Crowd counting | > 50 people |

### Face Analysis
| Model | HuggingFace ID | Use Case | Accuracy |
|-------|----------------|----------|----------|
| RetinaFace | `biubug6/insightface` | Face detection | 95%+ |
| FairFace | `dima806/fairface_gender_age` | Age + Gender | 85-95% |
| ViT-Age | `nateraw/vit-age-classifier` | Age only | 85% |
| ArcFace | `deepinsight/arcface` | Face re-ID | 99%+ |
| Emotion | `trpakov/vit-face-expression` | Facial emotion | 65-70% |

### Person Re-Identification
| Model | HuggingFace ID | Use Case | mAP |
|-------|----------------|----------|-----|
| OSNet | `KaiyangZhou/deep-person-reid` | General ReID | 85%+ |
| DINO | `facebook/dino-vitb16` | Self-supervised | 80%+ |
| CLIP | `openai/clip-vit-base-patch32` | Zero-shot ReID | 75%+ |

### Behavior & Pose
| Model | HuggingFace ID | Use Case |
|-------|----------------|----------|
| MediaPipe | (Google, not HF) | Real-time pose |
| ViTPose | `nielsr/vitpose-base` | High accuracy pose |
| SlowFast | `facebook/slowfast` | Action recognition |
| CLIP | `openai/clip-vit-base-patch32` | Activity classification |

### Scene Understanding
| Model | HuggingFace ID | Use Case |
|-------|----------------|----------|
| CLIP | `openai/clip-vit-base-patch32` | Zero-shot scene |
| Places365 | `nateraw/places365` | Venue classification |
| SegFormer | `nvidia/segformer-b0-finetuned-ade-512-512` | Scene segmentation |

---

## Mass Video Upload System

### Upload Methods

```
1. WEB INTERFACE (Single/Batch)
   ┌─────────────────────────────────────────────┐
   │  📁 Drop videos here or click to browse     │
   │  ─────────────────────────────────────────  │
   │  [Select 100+ files at once]                │
   │                                              │
   │  Location: ○ Auto-detect from GPS           │
   │            ○ Enter manually                  │
   │            ○ Select on map                   │
   │            ○ Assign to existing venue        │
   │                                              │
   │  Processing: ○ Full analytics               │
   │              ○ Count only (faster)           │
   │              ○ Demographics only             │
   └─────────────────────────────────────────────┘

2. API UPLOAD (Programmatic)
   POST /api/v1/videos/batch
   {
     "videos": [
       {
         "url": "s3://bucket/video1.mp4",
         "venue_id": "venue_123",
         "lat": -26.2041,
         "long": 28.0473,
         "captured_at": "2026-01-30T10:00:00Z"
       },
       ...
     ],
     "processing_level": "full",
     "priority": "normal"
   }

3. EDGE DEVICE SYNC
   - Cameras push directly to cloud
   - Automatic GPS from device
   - Real-time processing

4. FOLDER WATCH
   - Monitor S3/GCS bucket
   - Auto-process new uploads
   - Extract metadata from filename pattern:
     venue123_20260130_1000_-26.2041_28.0473.mp4
```

### Location Input Methods

```
Priority Order:
1. GPS from video EXIF metadata (automatic)
2. GPS from mobile upload (automatic)
3. Venue ID lookup (existing venues)
4. Filename pattern parsing
5. Map pin drop (manual)
6. Address geocoding (manual)
7. Default venue location (fallback)

Location Enrichment:
- Reverse geocode to get address
- Assign to H3 hexagon zone
- Lookup zone affluence data
- Get timezone for time normalization
```

---

## Zone Intelligence (Uber/Deliveroo Style)

### H3 Hexagonal Grid System

```
Why H3 (Uber's system)?
- Consistent hexagon sizes
- No edge effects like square grids
- Hierarchical (zoom in/out)
- Standard across the globe

Resolution Levels:
- Res 4: ~1,770 km² (country level)
- Res 6: ~36 km² (city level)
- Res 8: ~0.74 km² (district level)
- Res 9: ~0.1 km² (micro-zone - our primary)
- Res 11: ~0.003 km² (street level)
```

### Zone Pricing/Weighting (like Uber surge)

```python
class ZoneIntelligence:
    def calculate_zone_score(self, zone_id):
        """
        Like Uber's surge pricing, but for analytics value
        """
        factors = {
            "traffic_volume": self.get_traffic(zone_id),      # More traffic = more valuable
            "affluence_index": self.get_affluence(zone_id),   # Wealthier = higher spend
            "business_density": self.get_businesses(zone_id), # More venues = more data
            "competition": self.get_competitors(zone_id),     # Less competition = opportunity
            "growth_rate": self.get_growth(zone_id),          # Growing areas = future value
        }

        return weighted_score(factors)

    def get_zone_insights(self, zone_id):
        return {
            "score": self.calculate_zone_score(zone_id),
            "recommendation": "High potential zone - prioritize venue acquisition",
            "similar_zones": self.find_similar(zone_id),
            "benchmark": self.get_benchmark(zone_id),
        }
```

### Affluence Data Sources (Africa-specific)

```
Free/Low-cost:
- OpenStreetMap building data
- Facebook population density maps
- WorldPop population estimates
- Nighttime lights satellite data (proxy for development)
- Mobile coverage maps

Paid/Premium:
- MasterCard spending data
- Visa transaction insights
- Local census data
- Property value databases
- Retail sales data
```

---

## Dynamic Model Routing

```python
class ModelRouter:
    """
    Automatically select the best model combination
    based on input characteristics and requirements
    """

    def __init__(self):
        self.models = {
            "detection": {
                "fast": "yolov8n",      # < 10 people, edge
                "balanced": "yolov8m",   # 10-50 people
                "accurate": "yolov8l",   # > 50 or critical
                "crowd": "csrnet",       # Dense crowds
            },
            "tracking": {
                "fast": "bytetrack",
                "balanced": "deepsort",
                "accurate": "strongsort",
            },
            "demographics": {
                "fast": "vit-age-gender",
                "accurate": "fairface",
                "detailed": "insightface",
            },
            "behavior": {
                "fast": "mediapipe",
                "accurate": "vitpose",
                "action": "slowfast",
            }
        }

    def route(self, video_analysis):
        """Select models based on video characteristics"""

        config = {}

        # Detection model
        if video_analysis.estimated_people > 50:
            config["detection"] = "crowd"
        elif video_analysis.estimated_people > 10:
            config["detection"] = "balanced"
        elif video_analysis.is_edge_device:
            config["detection"] = "fast"
        else:
            config["detection"] = "accurate"

        # Demographics - only if faces visible
        if video_analysis.face_visibility > 0.3:
            if video_analysis.requires_ethnicity:
                config["demographics"] = "detailed"  # InsightFace
            else:
                config["demographics"] = "accurate"  # FairFace

        # Behavior - only if requested
        if video_analysis.requires_behavior:
            if video_analysis.is_edge_device:
                config["behavior"] = "fast"  # MediaPipe
            else:
                config["behavior"] = "accurate"  # ViTPose

        return config
```

---

## Map Insights & Filters

### Filter Options

```
TIME FILTERS:
├── Last hour / 6 hours / 24 hours
├── Last 7 days / 30 days / 90 days
├── Custom date range
├── Day of week comparison
├── Hour of day comparison
└── YoY / MoM comparison

DEMOGRAPHIC FILTERS:
├── Age groups (18-24, 25-34, 35-44, 45+)
├── Gender (if detected)
├── Group size (solo, couples, groups)
└── Visitor type (new vs returning)

BEHAVIOR FILTERS:
├── Dwell time (< 5min, 5-15min, > 15min)
├── Engagement level (browsing, engaged, purchasing)
├── Path patterns (entry→exit flow)
└── Peak hour activity

LOCATION FILTERS:
├── Country
├── City
├── District type (CBD, suburban, township)
├── Affluence level (high, medium, low)
├── Venue type (bar, restaurant, retail)
└── Custom zone groups
```

### Insight Cards (Auto-generated)

```
┌─────────────────────────────────────────────────────────────┐
│ 📈 GROWTH ALERT                                             │
│                                                              │
│ Sandton zone showing +23% traffic vs last month             │
│ Demographics: 25-34 age group driving growth                │
│ Peak shift: Now 8pm vs 7pm previously                       │
│                                                              │
│ Recommendation: Extend happy hour to 9pm                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 🎯 OPPORTUNITY                                               │
│                                                              │
│ Rosebank zone: High affluence, low venue coverage           │
│ Similar to Sandton profile but 40% fewer venues             │
│ Competitor gap: No sports bars detected                     │
│                                                              │
│ Recommendation: Expand into this zone                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ ⚠️ ANOMALY                                                   │
│                                                              │
│ CBD zone: Traffic down 35% this week                        │
│ Not explained by weather or holidays                        │
│ Competitor venue opened nearby                              │
│                                                              │
│ Recommendation: Investigate and adjust                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Core Analytics ✓ (DONE)
- Person detection + tracking
- Accurate counting with confidence
- Basic dwell time
- Zone assignment within venue

### Phase 2: Demographics (2-3 weeks)
- FairFace integration
- Real age/gender detection
- Face detection + crop
- Aggregate privacy (no individual storage)

### Phase 3: Geo-Location (2-3 weeks)
- Lat/long capture from uploads
- H3 zone assignment
- Zone management UI
- Basic map visualization

### Phase 4: Mass Upload (1-2 weeks)
- Batch upload interface
- Processing queue
- Status tracking
- Automatic metadata extraction

### Phase 5: Map Dashboard (3-4 weeks)
- Interactive map (Mapbox/Leaflet)
- Zone drill-down
- Filter system
- Insight cards

### Phase 6: Model Router (2-3 weeks)
- Automatic model selection
- Crowd density detection
- Quality-based routing
- Performance optimization

### Phase 7: Zone Intelligence (3-4 weeks)
- Affluence data integration
- Competition mapping
- Benchmark system
- Recommendations engine

### Phase 8: Enterprise Features (4-6 weeks)
- Multi-tenant architecture
- API rate limiting
- Audit logging
- Compliance (GDPR/POPIA)

---

## Database Schema Additions

```sql
-- Venues with geo
CREATE TABLE venues (
    id UUID PRIMARY KEY,
    name VARCHAR(200),
    lat DECIMAL(10, 8),
    long DECIMAL(11, 8),
    h3_zone_res9 VARCHAR(20),  -- H3 index
    address TEXT,
    city VARCHAR(100),
    country VARCHAR(100),
    venue_type VARCHAR(50),
    affluence_score DECIMAL(3, 2),
    created_at TIMESTAMP
);

-- Zone metadata
CREATE TABLE zones (
    h3_index VARCHAR(20) PRIMARY KEY,
    resolution INT,
    parent_h3 VARCHAR(20),
    city VARCHAR(100),
    country VARCHAR(100),
    affluence_score DECIMAL(3, 2),
    population_density INT,
    business_count INT,
    avg_traffic_daily INT,
    updated_at TIMESTAMP
);

-- Processing jobs
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY,
    video_url TEXT,
    venue_id UUID REFERENCES venues(id),
    status VARCHAR(20),
    model_config JSONB,  -- Which models were used
    results JSONB,
    processing_time_ms INT,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Zone aggregates (materialized for speed)
CREATE TABLE zone_daily_stats (
    h3_index VARCHAR(20),
    date DATE,
    total_visitors INT,
    unique_visitors INT,
    avg_dwell_seconds DECIMAL(10, 2),
    demographics JSONB,
    peak_hour INT,
    venue_count INT,
    PRIMARY KEY (h3_index, date)
);
```

---

## API Endpoints

```
# Venues
POST   /api/v1/venues                    # Create venue with lat/long
GET    /api/v1/venues                    # List venues
GET    /api/v1/venues/:id                # Get venue details
GET    /api/v1/venues/:id/analytics      # Get venue analytics

# Zones
GET    /api/v1/zones                     # List zones with stats
GET    /api/v1/zones/:h3_index           # Get zone details
GET    /api/v1/zones/:h3_index/venues    # Get venues in zone
GET    /api/v1/zones/:h3_index/analytics # Get zone analytics
GET    /api/v1/zones/:h3_index/benchmark # Compare to similar zones

# Video Processing
POST   /api/v1/videos                    # Upload single video
POST   /api/v1/videos/batch              # Upload batch
GET    /api/v1/videos/:id/status         # Check processing status
GET    /api/v1/videos/:id/results        # Get results

# Map Data
GET    /api/v1/map/zones                 # Get all zones for map
GET    /api/v1/map/heatmap               # Get traffic heatmap data
GET    /api/v1/map/insights              # Get auto-generated insights

# Analytics
GET    /api/v1/analytics/overview        # Platform-wide stats
GET    /api/v1/analytics/trends          # Trend analysis
GET    /api/v1/analytics/compare         # Compare zones/venues
```

---

## Summary: What We Need

| Component | HuggingFace Model | Priority |
|-----------|-------------------|----------|
| Person Detection | ultralytics/yolov8 | ✓ Done |
| Tracking | DeepSORT | ✓ Done |
| Age/Gender | dima806/fairface | Phase 2 |
| Face Detection | insightface | Phase 2 |
| Crowd Counting | csrnet | Phase 6 |
| Scene Type | openai/clip | Phase 6 |
| Pose/Behavior | vitpose | Phase 7 |

| Component | External Service | Priority |
|-----------|------------------|----------|
| Maps | Mapbox/Leaflet | Phase 5 |
| Geocoding | Nominatim/Google | Phase 3 |
| H3 Zones | h3-py library | Phase 3 |
| Affluence Data | WorldPop/OSM | Phase 7 |

This gives you a Nielsen-quality analytics platform with Uber-style zone intelligence, specifically designed for African markets.
