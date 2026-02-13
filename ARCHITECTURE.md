# CCTV Analytics MVP — System Architecture

## End-to-End Flow

```
VIDEO INPUT                    ML PIPELINE                         OUTPUT
============                   ===========                         ======

YouTube URL ──┐                                                ┌── Events DB
              ├──► Frame       ┌─► YOLO11n ──► BoT-SORT ──┐   ├── Visitor Embeddings DB
File Upload ──┘    Extraction  │   Detection    Tracking   │   ├── Analytics API (JSON)
                   (OpenCV)    │                           │   ├── HTML Dashboards
                       │       ├─► InsightFace ──────────┐ │   └── CSV/JSON Export
                       │       │   SCRFD + ArcFace       │ │
                       ▼       │   (Face, Age, Gender,   │ │
                  Every 0.5s   │    512-dim Embedding)   │ │
                  sample frame │                         │ │
                       │       ├─► YOLO11-Pose ────────┐ │ │
                       │       │   (17 Keypoints,      │ │ │
                       ▼       │    Orientation,       │ │ │
                  ┌────────┐   │    Posture, Engage)   │ │ │
                  │ Frame  │───┘                       │ │ │
                  └────────┘                           ▼ ▼ ▼
                                                  ┌──────────┐
                                                  │ Per-Track│
                                                  │ Aggregate│
                                                  └────┬─────┘
                                                       │
                                                       ▼
                                                ┌────────────┐
                                                │ Dedup +    │
                                                │ Concurrent │
                                                │ Cap Filter │
                                                └─────┬──────┘
                                                      │
                                                      ▼
                                                ┌───────────┐
                                                │ ReID Match│
                                                │ + Event   │
                                                │ Generation│
                                                └─────┬─────┘
                                                      │
                                                      ▼
                                                  DATABASE
```

---

## Stage-by-Stage Breakdown

### 1. Video Input

| Method | Endpoint | Handler |
|--------|----------|---------|
| YouTube URL | `POST /process/youtube` | `video_processing.py` → yt-dlp download |
| File Upload | `POST /process/upload` | `video_processing.py` → stream to temp |
| Batch URLs | `POST /api/batch/url` | `batch.py` → queued processing |
| Batch Upload | `POST /api/batch/upload` | `batch.py` → queued processing |

Each creates a `job_id` tracked in-memory and in the `jobs` DB table.

---

### 2. Frame Extraction (`pipeline.py`)

```
cv2.VideoCapture(video_path)
    │
    ├── Read total frame count + FPS
    ├── Calculate frame_interval = fps / 2  (sample every 0.5s)
    └── Loop: read frame every frame_interval
              ↓
         BGR frame (numpy array)
```

---

### 3. Detection + Tracking

```
Frame ──► YOLO11n(frame, classes=[0], conf=0.5)
              │
              ├── Detections: [bbox, confidence, class=person]
              │
              └──► BoT-SORT Tracker (botsort.yaml config)
                       │
                       ├── Assigns persistent track_id per person
                       ├── Maintains ID across frames (150-frame buffer)
                       └── Output: [track_id, x1, y1, x2, y2, conf]
```

**Models:** `yolo11n.pt` (ultralytics), BoT-SORT (built-in tracker)

---

### 4. Per-Person Crop Processing

For each detected person, extract crop `frame[y1:y2, x1:x2]` and run 3 parallel analyses:

#### 4a. Demographics (InsightFace buffalo_l)
```
Person Crop ──► SCRFD Face Detection
                    │
                    └──► Age (float → bracket) + Gender (M/F)

Age brackets: 0-17, 18-24, 25-34, 35-44, 45-54, 55-64, 60+
```

#### 4b. Face Embedding / ReID (InsightFace ArcFace)
```
Person Crop ──► SCRFD Face Detection
                    │
                    └──► ArcFace 512-dim Embedding (L2-normalized)
                         + quality_score (detection confidence)
```

#### 4c. Behavior Analysis (YOLO11-Pose)
```
Person Crop ──► YOLO11-Pose
                    │
                    ├── 17 COCO Keypoints (x, y, confidence)
                    │   nose, eyes, ears, shoulders, elbows,
                    │   wrists, hips, knees, ankles
                    │
                    ├──► body_orientation (-1 to +1)
                    │    shoulder visibility + horizontal spread
                    │
                    ├──► posture classification
                    │    upright | leaning_forward | leaning_back |
                    │    arms_crossed | hands_on_hips
                    │
                    ├──► movement estimation
                    │    stationary (<10 px/s) vs moving
                    │
                    ├──► engagement_score (0-100)
                    │    = 50 + orientation*20 + posture*15 + stationary*15
                    │
                    └──► behavior_type
                         engaged | browsing | waiting | passing
```

---

### 5. Track Aggregation

All per-frame data accumulated per `track_id`:

```
track_data[track_id] = {
    first_frame, last_frame,
    frame_count,
    zones: [list of grid zones visited],
    bboxes: [per-frame bounding boxes],
    confidences: [per-frame detection conf],
    demographics: {age_bracket, gender},     ← best-confidence face
    embedding: numpy(512,),                   ← best-quality face
    embedding_quality: float,
    behaviors: [list of BehaviorResults],
}
```

---

### 6. Deduplication (Two Layers)

```
Layer 1: Embedding Dedup
──────────────────────────
For each pair of non-overlapping tracks:
  if cosine_similarity(emb_A, emb_B) >= 0.60:
    merge tracks (union-find)
    keep longest track as primary

Layer 2: Max-Concurrent Cap
──────────────────────────
Sweep-line algorithm:
  Find peak simultaneous track count (N)
  Keep only top N tracks by frame_count
  Discard the rest as fragments
```

**Why:** BoT-SORT can fragment one person into multiple track IDs after occlusion. These two filters reduce 8 → 5 on the test video (exact ground truth).

---

### 7. ReID Matching + Event Generation

```
For each final track:
    │
    ├── Has embedding? ──► VisitorMatcher.find_match()
    │   │                      │
    │   │   cosine_sim >= 0.68 ├──► RETURN VISITOR
    │   │                      │    visitor_id = matched_id
    │   │                      │    is_repeat = True
    │   │                      │    update visit_count in DB
    │   │                      │
    │   │   cosine_sim < 0.68  └──► NEW VISITOR
    │   │                           visitor_id = MD5(embedding)[:16]
    │   │                           save embedding to DB
    │   │
    │   └── No embedding ──► pseudo_id = MD5(track_id + date)
    │
    └──► INSERT INTO events (
            venue_id, pseudo_id, timestamp, zone,
            dwell_seconds, age_bracket, gender, is_repeat,
            track_frames, detection_conf,
            engagement_score, behavior_type,
            body_orientation, posture
         )
         + zone transition events for multi-zone visits
```

---

### 8. Database Schema

```
┌─────────────────┐     ┌──────────────────────┐
│     venues       │     │  visitor_embeddings   │
├─────────────────┤     ├──────────────────────┤
│ id (PK)         │◄────│ venue_id (FK)        │
│ name            │     │ visitor_id (unique)   │
│ api_key         │     │ embedding (BLOB 512d) │
│ lat, lon, h3    │     │ quality_score         │
│ city, country   │     │ visit_count           │
│ venue_type      │     │ first_seen, last_seen │
└────────┬────────┘     │ age_bracket, gender   │
         │              └──────────────────────┘
         │
         │  ┌──────────────────┐     ┌─────────────────┐
         ├──│      events       │     │      alerts      │
         │  ├──────────────────┤     ├─────────────────┤
         │  │ venue_id (FK)    │     │ venue_id (FK)   │
         │  │ pseudo_id        │     │ alert_type      │
         │  │ timestamp        │     │ severity        │
         │  │ zone             │     │ title, message  │
         │  │ dwell_seconds    │     │ data (JSON)     │
         │  │ age_bracket      │     │ acknowledged    │
         │  │ gender           │     └─────────────────┘
         │  │ is_repeat        │
         │  │ engagement_score │
         │  │ behavior_type    │     ┌─────────────────┐
         │  │ body_orientation │     │      jobs        │
         │  │ posture          │     ├─────────────────┤
         │  └──────────────────┘     │ venue_id (FK)   │
         │                           │ status          │
         └───────────────────────────│ video_source    │
                                     │ progress        │
                                     │ visitors_detected│
                                     └─────────────────┘
```

---

### 9. Analytics Output Layer

| Endpoint | What It Queries | Output |
|----------|----------------|--------|
| `GET /analytics/{venue_id}` | events aggregate | visitors, dwell, return rate |
| `GET /analytics/{venue_id}/hourly` | events by hour | 24-hour traffic pattern |
| `GET /api/summary/{venue_id}` | events + embeddings | executive KPIs |
| `GET /api/demographics/{venue_id}` | events.age_bracket, gender | age/gender breakdown |
| `GET /api/zones/{venue_id}` | events.zone | zone traffic + dwell |
| `GET /api/behavior/{venue_id}` | events.behavior_type | engagement analysis |
| `GET /api/visitors/{venue_id}` | visitor_embeddings | known visitor list |
| `GET /api/heatmap/{venue_id}` | events by hour+day | weekly activity heatmap |
| `GET /api/benchmark/venues` | events cross-venue | venue comparison |

---

## ML Model Stack

| Model | Size | Framework | Purpose |
|-------|------|-----------|---------|
| **YOLO11n** | 5.4MB | ultralytics | Person detection |
| **BoT-SORT** | — | ultralytics (built-in) | Multi-object tracking |
| **buffalo_l (SCRFD)** | 30MB | InsightFace/ONNX | Face detection |
| **buffalo_l (ArcFace)** | 250MB | InsightFace/ONNX | Face embedding (512-d) |
| **buffalo_l (age/gender)** | included | InsightFace/ONNX | Demographics |
| **YOLO11n-pose** | 5.6MB | ultralytics | Body pose (17 keypoints) |

All models pre-loaded at startup via background thread (`models.py`).

---

## Key Thresholds

| Parameter | Value | Effect |
|-----------|-------|--------|
| ReID match threshold | 0.68 | Higher = fewer false return-visitor matches |
| Dedup merge threshold | 0.60 | Lower than ReID — merges obvious same-person fragments |
| Min track frames | 8 | Filters brief false detections |
| Detection confidence | 0.50 | YOLO person detection cutoff |
| Frame sampling | every 0.5s | Balances accuracy vs speed |
| BoT-SORT track buffer | 150 frames | How long to remember lost tracks |
