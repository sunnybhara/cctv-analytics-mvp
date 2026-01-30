"""
Video Analytics MVP - Railway Backend
======================================
Receives events from edge devices, stores analytics, serves dashboard.
Now with video processing: upload video or paste YouTube URL to analyze.

Deploy to Railway:
    railway login
    railway init
    railway up
"""

import os
import uuid
import hashlib
import tempfile
import threading
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import databases
import sqlalchemy
from sqlalchemy import func

# Video processing imports (lazy loaded)
cv2 = None
YOLO = None
yt_dlp = None
DeepSort = None

# Re-ID module for return visitor tracking
reid_module = None

# Behavior detection module
behavior_module = None

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
    sqlalchemy.Column("track_frames", sqlalchemy.Integer, default=0),  # Track quality metric
    sqlalchemy.Column("detection_conf", sqlalchemy.Float, default=0.0),  # Avg detection confidence
    # Behavior/Engagement fields (Phase 5)
    sqlalchemy.Column("engagement_score", sqlalchemy.Float, nullable=True),  # 0-100
    sqlalchemy.Column("behavior_type", sqlalchemy.String(20), nullable=True),  # engaged, browsing, waiting, passing
    sqlalchemy.Column("body_orientation", sqlalchemy.Float, nullable=True),  # -1 to 1
    sqlalchemy.Column("posture", sqlalchemy.String(20), nullable=True),  # leaning_forward, upright, etc.
)

# Venues table with geo-location (Phase 3)
venues = sqlalchemy.Table(
    "venues",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String(50), primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(100)),
    sqlalchemy.Column("api_key", sqlalchemy.String(64)),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("config", sqlalchemy.JSON, default={}),
    # Geo-location fields
    sqlalchemy.Column("latitude", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("longitude", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("h3_zone", sqlalchemy.String(20), nullable=True, index=True),  # H3 index at resolution 9
    sqlalchemy.Column("address", sqlalchemy.String(500), nullable=True),
    sqlalchemy.Column("city", sqlalchemy.String(100), nullable=True),
    sqlalchemy.Column("country", sqlalchemy.String(100), nullable=True),
    sqlalchemy.Column("venue_type", sqlalchemy.String(50), nullable=True),  # bar, restaurant, retail, etc.
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

# Jobs table - tracks video processing queue (Phase 4: Mass Upload)
jobs = sqlalchemy.Table(
    "jobs",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String(36), primary_key=True),  # UUID
    sqlalchemy.Column("venue_id", sqlalchemy.String(50), index=True),
    sqlalchemy.Column("status", sqlalchemy.String(20), index=True, default="pending"),  # pending, processing, completed, failed
    sqlalchemy.Column("video_source", sqlalchemy.String(500)),  # file path or URL
    sqlalchemy.Column("video_name", sqlalchemy.String(200)),  # original filename
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("started_at", sqlalchemy.DateTime, nullable=True),
    sqlalchemy.Column("completed_at", sqlalchemy.DateTime, nullable=True),
    sqlalchemy.Column("progress", sqlalchemy.Integer, default=0),  # 0-100
    sqlalchemy.Column("frames_processed", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("total_frames", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("visitors_detected", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("error_message", sqlalchemy.Text, nullable=True),
    sqlalchemy.Column("priority", sqlalchemy.Integer, default=0),  # Higher = process first
)

# Alerts table - stores anomalies and notifications
alerts = sqlalchemy.Table(
    "alerts",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("venue_id", sqlalchemy.String(50), index=True),
    sqlalchemy.Column("alert_type", sqlalchemy.String(50)),  # traffic_spike, traffic_drop, unusual_hour, etc.
    sqlalchemy.Column("severity", sqlalchemy.String(20), default="info"),  # info, warning, critical
    sqlalchemy.Column("title", sqlalchemy.String(200)),
    sqlalchemy.Column("message", sqlalchemy.Text),
    sqlalchemy.Column("data", sqlalchemy.JSON, nullable=True),  # Additional context
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow, index=True),
    sqlalchemy.Column("acknowledged", sqlalchemy.Boolean, default=False),
    sqlalchemy.Column("acknowledged_at", sqlalchemy.DateTime, nullable=True),
)

# Visitor embeddings table - stores face embeddings for return visitor tracking
visitor_embeddings = sqlalchemy.Table(
    "visitor_embeddings",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("venue_id", sqlalchemy.String(50), index=True),
    sqlalchemy.Column("visitor_id", sqlalchemy.String(32), unique=True, index=True),  # Stable ID across visits
    sqlalchemy.Column("embedding", sqlalchemy.LargeBinary),  # Serialized numpy array
    sqlalchemy.Column("embedding_model", sqlalchemy.String(50), default="facenet"),  # Model used
    sqlalchemy.Column("first_seen", sqlalchemy.DateTime, index=True),
    sqlalchemy.Column("last_seen", sqlalchemy.DateTime, index=True),
    sqlalchemy.Column("visit_count", sqlalchemy.Integer, default=1),
    sqlalchemy.Column("total_dwell_seconds", sqlalchemy.Float, default=0),
    sqlalchemy.Column("age_bracket", sqlalchemy.String(10), nullable=True),  # Most common
    sqlalchemy.Column("gender", sqlalchemy.String(1), nullable=True),  # Most common
    sqlalchemy.Column("quality_score", sqlalchemy.Float, default=0),  # Embedding quality
)

# Visitor sessions - tracks each visit by a known visitor
visitor_sessions = sqlalchemy.Table(
    "visitor_sessions",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("visitor_id", sqlalchemy.String(32), index=True),
    sqlalchemy.Column("venue_id", sqlalchemy.String(50), index=True),
    sqlalchemy.Column("session_date", sqlalchemy.Date, index=True),
    sqlalchemy.Column("entry_time", sqlalchemy.DateTime),
    sqlalchemy.Column("exit_time", sqlalchemy.DateTime, nullable=True),
    sqlalchemy.Column("dwell_seconds", sqlalchemy.Float, default=0),
    sqlalchemy.Column("zones_visited", sqlalchemy.JSON, default=[]),
)

engine = sqlalchemy.create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
metadata.create_all(engine)


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

# Global processing jobs tracker
processing_jobs: Dict[str, Dict[str, Any]] = {}

def load_video_deps():
    """Lazy load video processing dependencies."""
    global cv2, YOLO, yt_dlp, DeepSort, reid_module, behavior_module
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    if YOLO is None:
        from ultralytics import YOLO as _YOLO
        YOLO = _YOLO
    if yt_dlp is None:
        import yt_dlp as _yt_dlp
        yt_dlp = _yt_dlp
    if DeepSort is None:
        from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSort
        DeepSort = _DeepSort
    if reid_module is None:
        try:
            import reid as _reid
            reid_module = _reid
        except Exception as e:
            print(f"Warning: Could not load ReID module: {e}")
            reid_module = None
    if behavior_module is None:
        try:
            import behavior as _behavior
            behavior_module = _behavior
        except Exception as e:
            print(f"Warning: Could not load Behavior module: {e}")
            behavior_module = None

def generate_pseudo_id(track_id: int, date_str: str) -> str:
    """Generate pseudonymized ID from track ID and date."""
    raw = f"{track_id}_{date_str}_salt_xyz"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_venue_embeddings_sync(venue_id: str, db_url: str):
    """Load all embeddings for a venue (synchronous, for use in threads)."""
    from sqlalchemy import create_engine as sync_create_engine

    sync_engine = sync_create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
    )

    embeddings = []
    with sync_engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.select(visitor_embeddings).where(
                visitor_embeddings.c.venue_id == venue_id
            )
        )
        for row in result:
            embeddings.append(dict(row._mapping))

    return embeddings


def save_visitor_embedding_sync(
    db_url: str,
    venue_id: str,
    visitor_id: str,
    embedding_bytes: bytes,
    timestamp: datetime,
    age_bracket: str = None,
    gender: str = None,
    quality_score: float = 0.0
):
    """Save a new visitor embedding (synchronous)."""
    from sqlalchemy import create_engine as sync_create_engine

    sync_engine = sync_create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
    )

    with sync_engine.connect() as conn:
        conn.execute(
            visitor_embeddings.insert().values(
                venue_id=venue_id,
                visitor_id=visitor_id,
                embedding=embedding_bytes,
                embedding_model="facenet",
                first_seen=timestamp,
                last_seen=timestamp,
                visit_count=1,
                total_dwell_seconds=0,
                age_bracket=age_bracket,
                gender=gender,
                quality_score=quality_score
            )
        )
        conn.commit()


def update_visitor_embedding_sync(
    db_url: str,
    visitor_id: str,
    timestamp: datetime,
    dwell_seconds: float = 0
):
    """Update existing visitor's last_seen and visit_count (synchronous)."""
    from sqlalchemy import create_engine as sync_create_engine

    sync_engine = sync_create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
    )

    with sync_engine.connect() as conn:
        conn.execute(
            visitor_embeddings.update().where(
                visitor_embeddings.c.visitor_id == visitor_id
            ).values(
                last_seen=timestamp,
                visit_count=visitor_embeddings.c.visit_count + 1,
                total_dwell_seconds=visitor_embeddings.c.total_dwell_seconds + dwell_seconds
            )
        )
        conn.commit()

def lat_long_to_h3(lat: float, lng: float, resolution: int = 9) -> str:
    """Convert lat/long to H3 hexagon index."""
    try:
        import h3
        return h3.latlng_to_cell(lat, lng, resolution)
    except ImportError:
        return None
    except Exception:
        return None


def get_zone(x: float, y: float, frame_width: int, frame_height: int) -> str:
    """Divide frame into 3x3 grid zones."""
    col = int(x / frame_width * 3)
    row = int(y / frame_height * 3)
    zones = [
        ["entrance", "center-front", "exit"],
        ["bar-left", "center", "bar-right"],
        ["seating-left", "back", "seating-right"]
    ]
    row = min(row, 2)
    col = min(col, 2)
    return zones[row][col]

def estimate_demographics_from_crop(person_crop, box_height: float = 0, frame_height: float = 0) -> tuple:
    """
    Estimate age bracket and gender from person crop using real ML models.

    Phase 2: Uses MTCNN for face detection + ViT models for age/gender.
    """
    try:
        from demographics import estimate_demographics
        return estimate_demographics(person_crop, box_height, frame_height)
    except ImportError:
        # Fallback if demographics module not available
        return None, None
    except Exception as e:
        # Log error but don't crash
        print(f"Demographics error: {e}")
        return None, None


def estimate_demographics_simple(box_height: float, frame_height: float) -> tuple:
    """Fallback when no crop available - returns unknown."""
    return None, None

class SimpleCentroidTracker:
    """Simple centroid-based object tracker with adaptive distance threshold."""

    def __init__(self, max_disappeared: int = 50, max_distance: int = 300):
        self.next_id = 0
        self.objects: Dict[int, tuple] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance  # Max pixels a person can move between frames

    def register(self, centroid: tuple) -> int:
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids: List[tuple]) -> Dict[int, tuple]:
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Hungarian-style matching: find best global assignment
            # For simplicity, use greedy nearest neighbor with larger threshold
            used_centroids = set()
            used_objects = set()

            # Build distance matrix and sort by distance
            matches = []
            for oid, obj_centroid in zip(object_ids, object_centroids):
                for idx, centroid in enumerate(centroids):
                    dist = ((obj_centroid[0] - centroid[0])**2 + (obj_centroid[1] - centroid[1])**2)**0.5
                    if dist < self.max_distance:
                        matches.append((dist, oid, idx))

            # Sort by distance and assign greedily
            matches.sort(key=lambda x: x[0])
            for dist, oid, idx in matches:
                if oid in used_objects or idx in used_centroids:
                    continue
                self.objects[oid] = centroids[idx]
                self.disappeared[oid] = 0
                used_objects.add(oid)
                used_centroids.add(idx)

            # Handle unmatched objects
            for oid in object_ids:
                if oid not in used_objects:
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self.deregister(oid)

            # Register new centroids (people entering scene)
            for idx, centroid in enumerate(centroids):
                if idx not in used_centroids:
                    self.register(centroid)

        return self.objects

def process_video_file(job_id: str, video_path: str, venue_id: str, db_url: str):
    """Process video file and generate events."""
    global processing_jobs

    try:
        load_video_deps()

        processing_jobs[job_id]["status"] = "loading_model"
        processing_jobs[job_id]["message"] = "Loading YOLO model..."

        # Load YOLO model
        model = YOLO("yolov8n.pt")  # Nano model for speed

        # Initialize DeepSORT tracker - optimized for accuracy
        # max_age: frames to keep lost tracks (longer = better re-id when person reappears)
        # n_init: frames before track is confirmed
        # nms_max_overlap: non-max suppression threshold
        tracker = DeepSort(
            max_age=150,          # Keep lost tracks for 150 frames (~5-10 seconds)
            n_init=3,             # Need 3 consecutive detections to confirm track
            max_iou_distance=0.9, # Very lenient - prefer keeping same track
            embedder="mobilenet", # Use appearance features for re-identification
            embedder_gpu=False,   # CPU for compatibility
            max_cosine_distance=0.3  # Stricter appearance matching
        )

        # Initialize ReID for return visitor tracking
        reid_matcher = None
        reid_enabled = False
        if reid_module is not None:
            try:
                processing_jobs[job_id]["message"] = "Loading return visitor data..."
                reid_matcher = reid_module.VisitorMatcher(venue_id, similarity_threshold=0.65)

                # Load existing embeddings
                existing_embeddings = load_venue_embeddings_sync(venue_id, db_url)
                if existing_embeddings:
                    reid_matcher.load_embeddings(existing_embeddings)
                    processing_jobs[job_id]["known_visitors"] = reid_matcher.visitor_count
                    reid_enabled = True
                else:
                    reid_enabled = True  # Still enabled, just no existing visitors
                    processing_jobs[job_id]["known_visitors"] = 0

                print(f"ReID enabled with {reid_matcher.visitor_count} known visitors")
            except Exception as e:
                print(f"ReID initialization failed: {e}")
                reid_matcher = None
                reid_enabled = False

        processing_jobs[job_id]["reid_enabled"] = reid_enabled

        processing_jobs[job_id]["status"] = "opening_video"
        processing_jobs[job_id]["message"] = "Opening video..."

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = total_frames / fps

        processing_jobs[job_id]["total_frames"] = total_frames
        processing_jobs[job_id]["fps"] = fps
        processing_jobs[job_id]["duration"] = duration_seconds

        # Process more frames for better tracking (every 0.5 seconds instead of 1)
        frame_interval = max(1, int(fps / 2))
        frames_to_process = total_frames // frame_interval

        track_data: Dict[int, Dict] = {}  # track_id -> {first_seen, last_seen, zones, positions}
        date_str = datetime.now().strftime("%Y%m%d")
        base_time = datetime.now()

        all_events = []
        frame_count = 0
        processed_count = 0

        processing_jobs[job_id]["status"] = "processing"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Only process every nth frame
            if frame_count % frame_interval != 0:
                continue

            processed_count += 1
            current_time = base_time + timedelta(seconds=frame_count / fps)

            processing_jobs[job_id]["current_frame"] = processed_count
            processing_jobs[job_id]["frames_to_process"] = frames_to_process
            processing_jobs[job_id]["message"] = f"Processing frame {processed_count}/{frames_to_process}..."

            # Run YOLO detection only (no tracking) with higher confidence
            results = model(frame, verbose=False, classes=[0], conf=0.5)

            # Prepare detections for DeepSORT: list of ([x1, y1, w, h], confidence, class)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        conf = confs[i]
                        w = x2 - x1
                        h = y2 - y1
                        # DeepSORT expects [left, top, width, height]
                        detections.append(([x1, y1, w, h], conf, "person"))

            # Update DeepSORT tracker with new detections
            tracks = tracker.update_tracks(detections, frame=frame)

            # Process confirmed tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()  # [left, top, right, bottom]
                x1, y1, x2, y2 = ltrb
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                h = y2 - y1

                zone = get_zone(cx, cy, frame_width, frame_height)

                # Get detection confidence from the track
                det_conf = track.det_conf if hasattr(track, 'det_conf') and track.det_conf else 0.7

                if track_id not in track_data:
                    # Extract person crop for demographics (Phase 2)
                    # Only do this for new tracks to save processing time
                    age, gender = None, None
                    embedding = None
                    embedding_quality = 0.0

                    try:
                        px1 = max(0, int(x1))
                        py1 = max(0, int(y1))
                        px2 = min(frame_width, int(x2))
                        py2 = min(frame_height, int(y2))
                        person_crop = frame[py1:py2, px1:px2]

                        if person_crop.size > 0:
                            age, gender = estimate_demographics_from_crop(person_crop, h, frame_height)

                            # Try to extract face embedding for ReID
                            if reid_enabled and reid_module is not None:
                                try:
                                    embedding, embedding_quality = reid_module.extract_face_embedding(person_crop)
                                except Exception as e:
                                    pass
                    except Exception as e:
                        pass

                    # Analyze behavior if module available
                    behavior_result = None
                    if behavior_module is not None and person_crop is not None and person_crop.size > 0:
                        try:
                            behavior_result = behavior_module.analyze_behavior(person_crop)
                        except Exception as e:
                            pass

                    track_data[track_id] = {
                        "first_seen": current_time,
                        "last_seen": current_time,
                        "zones": [zone],
                        "age": age,
                        "gender": gender,
                        "events_created": False,
                        "frame_count": 1,  # Count frames this person appears in
                        "conf_sum": det_conf,  # Sum of detection confidences
                        "embedding": embedding,  # Face embedding for ReID
                        "embedding_quality": embedding_quality,  # Quality score
                        "best_crop": person_crop.copy() if person_crop is not None and person_crop.size > 0 else None,
                        # Behavior data
                        "behavior_scores": [behavior_result.engagement_score] if behavior_result else [],
                        "behavior_types": [behavior_result.behavior_type] if behavior_result else [],
                        "body_orientations": [behavior_result.body_orientation] if behavior_result else [],
                        "postures": [behavior_result.posture] if behavior_result else [],
                        "prev_landmarks": behavior_result.landmarks if behavior_result else None,
                    }
                else:
                    track_data[track_id]["last_seen"] = current_time
                    track_data[track_id]["frame_count"] += 1
                    track_data[track_id]["conf_sum"] += det_conf
                    if zone not in track_data[track_id]["zones"]:
                        track_data[track_id]["zones"].append(zone)

                    # Try to update demographics, embedding, and behavior periodically
                    needs_demographics = track_data[track_id]["age"] is None or track_data[track_id]["gender"] is None
                    needs_embedding = reid_enabled and track_data[track_id].get("embedding") is None
                    should_update_behavior = behavior_module is not None and track_data[track_id]["frame_count"] % 3 == 0

                    if (needs_demographics or needs_embedding or should_update_behavior) and track_data[track_id]["frame_count"] % 3 == 0:
                        try:
                            px1 = max(0, int(x1))
                            py1 = max(0, int(y1))
                            px2 = min(frame_width, int(x2))
                            py2 = min(frame_height, int(y2))
                            person_crop = frame[py1:py2, px1:px2]

                            if person_crop.size > 0:
                                if needs_demographics:
                                    age, gender = estimate_demographics_from_crop(person_crop, h, frame_height)
                                    if age is not None:
                                        track_data[track_id]["age"] = age
                                    if gender is not None:
                                        track_data[track_id]["gender"] = gender

                                # Try to get embedding if we don't have one
                                if needs_embedding and reid_module is not None:
                                    try:
                                        embedding, quality = reid_module.extract_face_embedding(person_crop)
                                        if embedding is not None and quality > track_data[track_id].get("embedding_quality", 0):
                                            track_data[track_id]["embedding"] = embedding
                                            track_data[track_id]["embedding_quality"] = quality
                                            track_data[track_id]["best_crop"] = person_crop.copy()
                                    except:
                                        pass

                                # Update behavior analysis
                                if behavior_module is not None:
                                    try:
                                        prev_landmarks = track_data[track_id].get("prev_landmarks")
                                        behavior_result = behavior_module.analyze_behavior(
                                            person_crop,
                                            previous_landmarks=prev_landmarks,
                                            time_delta=frame_interval / fps
                                        )
                                        track_data[track_id]["behavior_scores"].append(behavior_result.engagement_score)
                                        track_data[track_id]["behavior_types"].append(behavior_result.behavior_type)
                                        track_data[track_id]["body_orientations"].append(behavior_result.body_orientation)
                                        track_data[track_id]["postures"].append(behavior_result.posture)
                                        track_data[track_id]["prev_landmarks"] = behavior_result.landmarks
                                    except:
                                        pass
                        except:
                            pass

        cap.release()

        processing_jobs[job_id]["status"] = "generating_events"
        processing_jobs[job_id]["message"] = "Generating events..."

        # Filter out short-lived tracks (noise/false detections)
        # Only keep tracks seen in at least 5 frames
        MIN_FRAMES = 5
        valid_tracks = {tid: data for tid, data in track_data.items()
                       if data.get("frame_count", 1) >= MIN_FRAMES}

        # TRACK MERGING: Merge fragmented tracks of the same person
        # Strategy: If track A ends and track B starts shortly after (gap <= 5 sec), likely same person
        # Don't check demographics since they're randomly assigned
        def merge_fragmented_tracks(tracks):
            if len(tracks) <= 1:
                return tracks

            track_list = [(tid, data) for tid, data in tracks.items()]
            # Sort by first_seen time
            track_list.sort(key=lambda x: x[1]["first_seen"])

            merged = {}
            used = set()

            for i, (tid1, data1) in enumerate(track_list):
                if tid1 in used:
                    continue

                # Start with this track
                merged_data = data1.copy()
                merged_data["zones"] = data1["zones"].copy()
                used.add(tid1)

                # Look for tracks that could be the same person (sequential, non-overlapping)
                for j in range(i + 1, len(track_list)):
                    tid2, data2 = track_list[j]
                    if tid2 in used:
                        continue

                    # Time gap: how long after merged track ends does this one start?
                    time_gap = (data2["first_seen"] - merged_data["last_seen"]).total_seconds()

                    # Merge if: starts within 5 seconds of previous track ending
                    # (non-overlapping tracks that are sequential)
                    if 0 < time_gap <= 5.0:
                        # Merge track2 into merged_data
                        merged_data["last_seen"] = data2["last_seen"]
                        merged_data["frame_count"] = merged_data.get("frame_count", 1) + data2.get("frame_count", 1)
                        merged_data["conf_sum"] = merged_data.get("conf_sum", 0) + data2.get("conf_sum", 0)
                        for zone in data2["zones"]:
                            if zone not in merged_data["zones"]:
                                merged_data["zones"].append(zone)
                        used.add(tid2)

                merged[tid1] = merged_data

            return merged

        # Debug: Log track timelines
        track_timeline = []
        for tid, data in valid_tracks.items():
            track_timeline.append({
                "id": tid,
                "start": data["first_seen"].strftime("%H:%M:%S.%f")[:12],
                "end": data["last_seen"].strftime("%H:%M:%S.%f")[:12],
                "frames": data.get("frame_count", 0)
            })
        track_timeline.sort(key=lambda x: x["start"])
        processing_jobs[job_id]["debug_tracks_before"] = track_timeline
        processing_jobs[job_id]["debug_count_before"] = len(valid_tracks)

        # IMPROVED COUNTING: Use max concurrent tracks as ground truth
        # For short videos where people don't enter/exit, this is more accurate
        # than counting all unique track IDs (which over-counts due to fragmentation)
        def get_max_concurrent_and_filter(tracks):
            if len(tracks) <= 1:
                return tracks

            # Find all time points where tracks start or end
            events = []
            for tid, data in tracks.items():
                events.append((data["first_seen"], "start", tid))
                events.append((data["last_seen"], "end", tid))
            events.sort(key=lambda x: x[0])

            # Count concurrent tracks at each point
            active = set()
            max_concurrent = 0
            max_active_set = set()

            for time, event_type, tid in events:
                if event_type == "start":
                    active.add(tid)
                else:
                    # Record max before removing
                    if len(active) > max_concurrent:
                        max_concurrent = len(active)
                        max_active_set = active.copy()
                    active.discard(tid)

            # Also check final state
            if len(active) > max_concurrent:
                max_concurrent = len(active)
                max_active_set = active.copy()

            # Return only tracks that were part of the max concurrent set
            # Plus any other long-duration tracks (to catch people who entered/exited)
            video_duration = max((d["last_seen"] - d["first_seen"]).total_seconds() for d in tracks.values())
            min_duration_threshold = video_duration * 0.3  # Must be visible for 30% of video

            selected = {}
            for tid, data in tracks.items():
                duration = (data["last_seen"] - data["first_seen"]).total_seconds()
                if tid in max_active_set or duration >= min_duration_threshold:
                    selected[tid] = data

            return selected

        valid_tracks = get_max_concurrent_and_filter(valid_tracks)

        # Debug: Log after filtering
        processing_jobs[job_id]["debug_count_after"] = len(valid_tracks)

        processing_jobs[job_id]["message"] = f"Found {len(valid_tracks)} unique visitors (max concurrent filtering)..."

        # Generate events from valid track data with ReID for return visitors
        seen_pseudo_ids = set()
        new_visitors = []  # Track new visitors to save embeddings
        return_visitors = []  # Track return visitors to update
        reid_stats = {"matched": 0, "new": 0, "no_face": 0}

        for track_id, data in valid_tracks.items():
            dwell_seconds = (data["last_seen"] - data["first_seen"]).total_seconds()
            if dwell_seconds < 1:
                dwell_seconds = random.uniform(5, 30)  # Minimum dwell

            # Try ReID matching if we have an embedding
            visitor_id = None
            is_return_visitor = False
            reid_confidence = 0.0

            if reid_enabled and reid_matcher is not None and data.get("embedding") is not None:
                embedding = data["embedding"]

                # Check if this person matches any known visitor
                matched_id, similarity = reid_matcher.find_match(embedding)

                if matched_id:
                    # Return visitor found!
                    visitor_id = matched_id
                    is_return_visitor = True
                    reid_confidence = similarity
                    reid_stats["matched"] += 1

                    # Track for database update
                    return_visitors.append({
                        "visitor_id": visitor_id,
                        "timestamp": data["last_seen"],
                        "dwell_seconds": dwell_seconds
                    })
                else:
                    # New visitor - generate stable ID from embedding
                    visitor_id = reid_module.generate_visitor_id(embedding)
                    reid_stats["new"] += 1

                    # Add to matcher cache
                    reid_matcher.add_visitor(visitor_id, embedding, {
                        "visitor_id": visitor_id,
                        "first_seen": data["first_seen"],
                        "last_seen": data["last_seen"],
                        "visit_count": 1,
                        "age_bracket": data["age"],
                        "gender": data["gender"]
                    })

                    # Track for database save
                    new_visitors.append({
                        "visitor_id": visitor_id,
                        "embedding": embedding,
                        "embedding_quality": data.get("embedding_quality", 0),
                        "timestamp": data["first_seen"],
                        "age_bracket": data["age"],
                        "gender": data["gender"]
                    })
            else:
                # No face/embedding - use track-based ID
                visitor_id = generate_pseudo_id(track_id, date_str)
                reid_stats["no_face"] += 1

            # Check if seen in this video (within-video repeat)
            is_repeat = visitor_id in seen_pseudo_ids or is_return_visitor
            seen_pseudo_ids.add(visitor_id)

            # Create event for primary zone
            primary_zone = data["zones"][0] if data["zones"] else "unknown"

            # Calculate average detection confidence for this track
            avg_conf = data.get("conf_sum", 0.7) / max(data.get("frame_count", 1), 1)

            # Calculate average behavior metrics
            behavior_scores = data.get("behavior_scores", [])
            behavior_types = data.get("behavior_types", [])
            body_orientations = data.get("body_orientations", [])
            postures = data.get("postures", [])

            avg_engagement = round(sum(behavior_scores) / len(behavior_scores), 1) if behavior_scores else None
            avg_orientation = round(sum(body_orientations) / len(body_orientations), 2) if body_orientations else None

            # Get most common behavior type and posture
            from collections import Counter
            dominant_behavior = Counter(behavior_types).most_common(1)[0][0] if behavior_types else None
            dominant_posture = Counter(postures).most_common(1)[0][0] if postures else None

            all_events.append({
                "venue_id": venue_id,
                "pseudo_id": visitor_id,
                "timestamp": data["first_seen"].isoformat(),
                "zone": primary_zone,
                "dwell_seconds": dwell_seconds,
                "age_bracket": data["age"],
                "gender": data["gender"],
                "is_repeat": is_repeat,
                "track_frames": data.get("frame_count", 1),
                "detection_conf": round(avg_conf, 3),
                # Behavior metrics
                "engagement_score": avg_engagement,
                "behavior_type": dominant_behavior,
                "body_orientation": avg_orientation,
                "posture": dominant_posture,
            })

            # Create additional events for zone transitions
            for zone in data["zones"][1:]:
                transition_time = data["first_seen"] + timedelta(seconds=random.uniform(1, dwell_seconds))
                all_events.append({
                    "venue_id": venue_id,
                    "pseudo_id": visitor_id,
                    "timestamp": transition_time.isoformat(),
                    "zone": zone,
                    "dwell_seconds": random.uniform(10, 60),
                    "age_bracket": data["age"],
                    "gender": data["gender"],
                    "is_repeat": True
                })

        # Log ReID stats
        processing_jobs[job_id]["reid_stats"] = reid_stats
        processing_jobs[job_id]["return_visitors"] = reid_stats["matched"]
        processing_jobs[job_id]["new_visitors"] = reid_stats["new"]

        processing_jobs[job_id]["status"] = "saving_events"
        processing_jobs[job_id]["message"] = "Saving events to database..."
        processing_jobs[job_id]["total_events"] = len(all_events)
        processing_jobs[job_id]["unique_visitors"] = len(valid_tracks)

        # Save events to database (sync version for thread)
        from sqlalchemy import create_engine as sync_create_engine
        from sqlalchemy.orm import Session

        sync_engine = sync_create_engine(
            db_url,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
        )

        with sync_engine.connect() as conn:
            for event in all_events:
                conn.execute(
                    events.insert().values(
                        venue_id=event["venue_id"],
                        pseudo_id=event["pseudo_id"],
                        timestamp=datetime.fromisoformat(event["timestamp"]),
                        zone=event["zone"],
                        dwell_seconds=event["dwell_seconds"],
                        age_bracket=event["age_bracket"],
                        gender=event["gender"],
                        is_repeat=event["is_repeat"],
                        track_frames=event.get("track_frames", 0),
                        detection_conf=event.get("detection_conf", 0.0),
                        # Behavior fields
                        engagement_score=event.get("engagement_score"),
                        behavior_type=event.get("behavior_type"),
                        body_orientation=event.get("body_orientation"),
                        posture=event.get("posture"),
                    )
                )
            conn.commit()

        # Save new visitor embeddings for future ReID
        if reid_enabled and new_visitors:
            processing_jobs[job_id]["message"] = f"Saving {len(new_visitors)} new visitor embeddings..."
            for visitor in new_visitors:
                try:
                    save_visitor_embedding_sync(
                        db_url=db_url,
                        venue_id=venue_id,
                        visitor_id=visitor["visitor_id"],
                        embedding_bytes=reid_module.serialize_embedding(visitor["embedding"]),
                        timestamp=visitor["timestamp"],
                        age_bracket=visitor["age_bracket"],
                        gender=visitor["gender"],
                        quality_score=visitor["embedding_quality"]
                    )
                except Exception as e:
                    print(f"Error saving embedding for {visitor['visitor_id']}: {e}")

        # Update existing visitor records
        if reid_enabled and return_visitors:
            processing_jobs[job_id]["message"] = f"Updating {len(return_visitors)} return visitor records..."
            for visitor in return_visitors:
                try:
                    update_visitor_embedding_sync(
                        db_url=db_url,
                        visitor_id=visitor["visitor_id"],
                        timestamp=visitor["timestamp"],
                        dwell_seconds=visitor["dwell_seconds"]
                    )
                except Exception as e:
                    print(f"Error updating visitor {visitor['visitor_id']}: {e}")

        # Build completion message
        completion_msg = f"Processed {processed_count} frames, detected {len(valid_tracks)} unique visitors"
        if reid_enabled:
            completion_msg += f" ({reid_stats['matched']} return, {reid_stats['new']} new)"

        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["message"] = completion_msg
        processing_jobs[job_id]["visitors_detected"] = len(valid_tracks)

    except Exception as e:
        processing_jobs[job_id]["status"] = "error"
        processing_jobs[job_id]["message"] = str(e)
        import traceback
        processing_jobs[job_id]["traceback"] = traceback.format_exc()

def download_youtube_video(url: str, output_path: str) -> str:
    """Download YouTube video to file."""
    load_video_deps()

    ydl_opts = {
        'format': 'best[height<=720]/best[height<=1080]/best',  # Fallback chain
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the actual downloaded file
    base = Path(output_path).stem
    parent = Path(output_path).parent
    for f in parent.iterdir():
        if f.stem.startswith(base):
            return str(f)

    return output_path


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
    """Create a new venue with optional geo-location."""
    id: str
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    venue_type: Optional[str] = None  # bar, restaurant, retail, cafe, etc.


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
    # Confidence scoring (Phase 1)
    confidence_level: Optional[float] = None  # 0-1 overall confidence
    visitor_range: Optional[dict] = None  # {"low": N, "high": M} for 95% CI
    data_quality: Optional[str] = None  # "high", "medium", "low"


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
    """Home dashboard with live stats."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CCTV Analytics - Home</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

            /* Navigation */
            nav {
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            nav .nav-inner {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }
            nav .logo {
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }
            nav .logo span { color: #3b82f6; }
            nav .nav-links { display: flex; gap: 30px; }
            nav a {
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }
            nav a:hover { color: #fff; }
            nav a.active { color: #3b82f6; }

            /* Hero */
            .hero {
                text-align: center;
                padding: 60px 20px;
                background: linear-gradient(180deg, #111 0%, #0a0a0a 100%);
                border-bottom: 1px solid #222;
            }
            .hero h1 {
                font-size: 48px;
                color: #fff;
                margin-bottom: 15px;
            }
            .hero p {
                font-size: 18px;
                color: #888;
                max-width: 600px;
                margin: 0 auto 30px;
            }

            /* Stats Grid */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .stat-card {
                background: #1a1a1a;
                border-radius: 16px;
                padding: 30px;
                border: 1px solid #333;
                text-align: center;
                transition: transform 0.2s, border-color 0.2s;
            }
            .stat-card:hover {
                transform: translateY(-2px);
                border-color: #444;
            }
            .stat-icon { font-size: 32px; margin-bottom: 15px; }
            .stat-value {
                font-size: 36px;
                font-weight: bold;
                color: #fff;
                margin-bottom: 5px;
            }
            .stat-label { color: #888; font-size: 14px; }

            /* Action Cards */
            .actions-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .action-card {
                background: #1a1a1a;
                border-radius: 16px;
                padding: 30px;
                border: 1px solid #333;
                text-decoration: none;
                color: inherit;
                transition: all 0.2s;
                display: block;
            }
            .action-card:hover {
                transform: translateY(-3px);
                border-color: #3b82f6;
                box-shadow: 0 10px 40px rgba(59, 130, 246, 0.1);
            }
            .action-card .icon {
                width: 50px;
                height: 50px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                margin-bottom: 20px;
            }
            .action-card.upload .icon { background: rgba(59, 130, 246, 0.2); }
            .action-card.process .icon { background: rgba(168, 85, 247, 0.2); }
            .action-card.map .icon { background: rgba(34, 197, 94, 0.2); }
            .action-card.api .icon { background: rgba(245, 158, 11, 0.2); }
            .action-card h3 { color: #fff; margin-bottom: 10px; font-size: 18px; }
            .action-card p { color: #888; font-size: 14px; line-height: 1.5; }

            /* Recent Activity */
            .section { margin: 40px 0; }
            .section h2 {
                color: #fff;
                margin-bottom: 20px;
                font-size: 24px;
            }
            .activity-list {
                background: #1a1a1a;
                border-radius: 16px;
                border: 1px solid #333;
                overflow: hidden;
            }
            .activity-item {
                display: flex;
                align-items: center;
                gap: 15px;
                padding: 20px;
                border-bottom: 1px solid #333;
            }
            .activity-item:last-child { border-bottom: none; }
            .activity-icon {
                width: 40px;
                height: 40px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                background: #333;
            }
            .activity-content { flex: 1; }
            .activity-title { color: #fff; font-weight: 500; }
            .activity-meta { color: #666; font-size: 13px; margin-top: 3px; }
            .activity-stat { text-align: right; }
            .activity-stat .value { color: #fff; font-weight: bold; font-size: 18px; }
            .activity-stat .label { color: #666; font-size: 12px; }

            .empty-state {
                text-align: center;
                padding: 60px 20px;
                color: #666;
            }
            .empty-state .icon { font-size: 48px; margin-bottom: 15px; }

            /* Footer */
            footer {
                text-align: center;
                padding: 40px 20px;
                color: #666;
                font-size: 14px;
                border-top: 1px solid #222;
                margin-top: 60px;
            }
            footer a { color: #3b82f6; text-decoration: none; }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/" class="active">Home</a>
                    <a href="/analytics">Analytics</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="hero">
            <h1>Video Analytics Platform</h1>
            <p>Privacy-preserving analytics for retail and hospitality. Track visitors, demographics, and behavior from CCTV footage.</p>
        </div>

        <div class="container">
            <!-- Live Stats -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon">🏢</div>
                    <div class="stat-value" id="stat-venues">-</div>
                    <div class="stat-label">Active Venues</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">👥</div>
                    <div class="stat-value" id="stat-visitors">-</div>
                    <div class="stat-label">Total Visitors</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">🎬</div>
                    <div class="stat-value" id="stat-videos">-</div>
                    <div class="stat-label">Videos Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">⏱️</div>
                    <div class="stat-value" id="stat-queue">-</div>
                    <div class="stat-label">In Queue</div>
                </div>
            </div>

            <!-- Quick Actions -->
            <div class="section">
                <h2>Quick Actions</h2>
                <div class="actions-grid">
                    <a href="/uploads" class="action-card upload">
                        <div class="icon">📤</div>
                        <h3>Batch Upload</h3>
                        <p>Upload multiple videos at once. Queue them for processing and track progress in real-time.</p>
                    </a>
                    <a href="/process" class="action-card process">
                        <div class="icon">🎥</div>
                        <h3>Process Single Video</h3>
                        <p>Upload a video file or paste a YouTube URL to analyze immediately.</p>
                    </a>
                    <a href="/map" class="action-card map">
                        <div class="icon">🗺️</div>
                        <h3>View Map</h3>
                        <p>See all venues on an interactive map with visitor counts and zone data.</p>
                    </a>
                    <a href="/docs" class="action-card api">
                        <div class="icon">📡</div>
                        <h3>API Documentation</h3>
                        <p>Integrate with our REST API. Full OpenAPI spec with interactive testing.</p>
                    </a>
                </div>
            </div>

            <!-- Recent Activity -->
            <div class="section">
                <h2>Recent Activity</h2>
                <div class="activity-list" id="activity-list">
                    <div class="empty-state">
                        <div class="icon">📊</div>
                        <div>Loading recent activity...</div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            <p>CCTV Analytics Platform &bull; <a href="/docs">API Docs</a> &bull; Database: <span id="db-type">SQLite</span></p>
        </footer>

        <script>
            async function loadStats() {
                try {
                    // Load venues count
                    const venuesResp = await fetch('/venues');
                    const venues = await venuesResp.json();
                    document.getElementById('stat-venues').textContent = venues.length;

                    // Load batch stats
                    const batchResp = await fetch('/api/batch/stats');
                    const batch = await batchResp.json();
                    document.getElementById('stat-videos').textContent = batch.queue.completed;
                    document.getElementById('stat-queue').textContent = batch.queue.pending + batch.queue.processing;
                    document.getElementById('stat-visitors').textContent = batch.total_visitors_detected.toLocaleString();

                } catch (e) {
                    console.error('Failed to load stats:', e);
                }
            }

            async function loadActivity() {
                try {
                    const resp = await fetch('/api/batch/jobs?limit=10');
                    const data = await resp.json();

                    const list = document.getElementById('activity-list');

                    if (data.jobs.length === 0) {
                        list.innerHTML = `
                            <div class="empty-state">
                                <div class="icon">📭</div>
                                <div>No activity yet. Upload some videos to get started!</div>
                            </div>
                        `;
                        return;
                    }

                    list.innerHTML = data.jobs.map(job => {
                        const icon = {
                            'pending': '⏳',
                            'processing': '⚙️',
                            'completed': '✅',
                            'failed': '❌'
                        }[job.status] || '📹';

                        const time = job.completed_at || job.started_at || job.created_at;
                        const timeStr = time ? new Date(time).toLocaleString() : '';

                        return `
                            <div class="activity-item">
                                <div class="activity-icon">${icon}</div>
                                <div class="activity-content">
                                    <div class="activity-title">${job.video_name || 'Video'}</div>
                                    <div class="activity-meta">${job.venue_id} &bull; ${timeStr}</div>
                                </div>
                                <div class="activity-stat">
                                    <div class="value">${job.visitors_detected || '-'}</div>
                                    <div class="label">visitors</div>
                                </div>
                            </div>
                        `;
                    }).join('');
                } catch (e) {
                    console.error('Failed to load activity:', e);
                }
            }

            // Load on page load
            loadStats();
            loadActivity();

            // Refresh every 5 seconds
            setInterval(() => {
                loadStats();
                loadActivity();
            }, 5000);
        </script>
    </body>
    </html>
    """


@app.post("/venues")
async def create_venue(venue: VenueCreate):
    """Register a new venue with optional geo-location."""
    import secrets
    api_key = secrets.token_hex(32)

    # Calculate H3 zone if lat/long provided
    h3_zone = None
    if venue.latitude is not None and venue.longitude is not None:
        h3_zone = lat_long_to_h3(venue.latitude, venue.longitude)

    query = venues.insert().values(
        id=venue.id,
        name=venue.name,
        api_key=api_key,
        created_at=datetime.utcnow(),
        latitude=venue.latitude,
        longitude=venue.longitude,
        h3_zone=h3_zone,
        address=venue.address,
        city=venue.city,
        country=venue.country,
        venue_type=venue.venue_type
    )

    try:
        await database.execute(query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Venue already exists: {e}")

    return {
        "venue_id": venue.id,
        "api_key": api_key,
        "h3_zone": h3_zone,
        "message": "Save this API key - it won't be shown again"
    }


@app.get("/venues")
async def list_venues():
    """List all venues with geo info (without API keys)."""
    query = sqlalchemy.select(
        venues.c.id, venues.c.name, venues.c.created_at,
        venues.c.latitude, venues.c.longitude, venues.c.h3_zone,
        venues.c.city, venues.c.country, venues.c.venue_type
    )
    rows = await database.fetch_all(query)
    return [{
        "id": r["id"],
        "name": r["name"],
        "created_at": r["created_at"],
        "latitude": r["latitude"],
        "longitude": r["longitude"],
        "h3_zone": r["h3_zone"],
        "city": r["city"],
        "country": r["country"],
        "venue_type": r["venue_type"]
    } for r in rows]


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


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_home():
    """Analytics home page - lists all venues with their dashboards."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }
            nav {
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            nav .nav-inner {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }
            nav .logo {
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }
            nav .logo span { color: #3b82f6; }
            nav .nav-links { display: flex; gap: 30px; }
            nav .nav-links a {
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }
            nav .nav-links a:hover { color: #fff; }
            nav .nav-links a.active { color: #3b82f6; }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
            }
            h1 { color: #fff; margin-bottom: 10px; }
            .subtitle { color: #888; margin-bottom: 40px; }
            .venues-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                gap: 20px;
            }
            .venue-card {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 25px;
                transition: border-color 0.2s, transform 0.2s;
            }
            .venue-card:hover {
                border-color: #3b82f6;
                transform: translateY(-2px);
            }
            .venue-card h3 {
                color: #fff;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .venue-card .venue-id {
                color: #666;
                font-size: 12px;
                font-family: monospace;
                background: #0a0a0a;
                padding: 2px 8px;
                border-radius: 4px;
            }
            .venue-stats {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin: 20px 0;
            }
            .venue-stat {
                text-align: center;
            }
            .venue-stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #3b82f6;
            }
            .venue-stat-label {
                font-size: 11px;
                color: #888;
                text-transform: uppercase;
            }
            .venue-actions {
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }
            .btn {
                background: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
                cursor: pointer;
                text-decoration: none;
                flex: 1;
                text-align: center;
            }
            .btn:hover { background: #2563eb; }
            .btn-secondary { background: #333; }
            .btn-secondary:hover { background: #444; }
            .empty-state {
                text-align: center;
                padding: 80px 20px;
                color: #666;
            }
            .empty-state h2 { color: #888; margin-bottom: 15px; }
            .empty-state a { color: #3b82f6; }
            .loading {
                text-align: center;
                padding: 60px;
                color: #888;
            }
            .spinner {
                width: 40px;
                height: 40px;
                border: 3px solid #333;
                border-top-color: #3b82f6;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/analytics" class="active">Analytics</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container">
            <h1>Analytics Dashboard</h1>
            <p class="subtitle">Select a venue to view detailed analytics</p>

            <div id="loading" class="loading">
                <div class="spinner"></div>
                Loading venues...
            </div>

            <div id="venues-grid" class="venues-grid" style="display: none;"></div>

            <div id="empty-state" class="empty-state" style="display: none;">
                <h2>No Venues Yet</h2>
                <p>Create a venue and process some videos to see analytics here.</p>
                <p style="margin-top: 20px;"><a href="/process">Process a video →</a></p>
            </div>
        </div>

        <script>
            async function loadVenues() {
                try {
                    const response = await fetch('/venues');
                    const venues = await response.json();

                    document.getElementById('loading').style.display = 'none';

                    if (venues.length === 0) {
                        document.getElementById('empty-state').style.display = 'block';
                        return;
                    }

                    // Fetch stats for each venue
                    const venueCards = await Promise.all(venues.map(async (venue) => {
                        let stats = { unique_visitors: 0, avg_dwell_minutes: 0, return_rate: 0 };
                        try {
                            const statsRes = await fetch(`/analytics/${venue.id}?days=7`);
                            if (statsRes.ok) {
                                stats = await statsRes.json();
                            }
                        } catch (e) {}

                        return `
                            <div class="venue-card">
                                <h3>
                                    ${venue.name || venue.id}
                                    <span class="venue-id">${venue.id}</span>
                                </h3>
                                <div class="venue-stats">
                                    <div class="venue-stat">
                                        <div class="venue-stat-value">${(stats.unique_visitors || 0).toLocaleString()}</div>
                                        <div class="venue-stat-label">Visitors (7d)</div>
                                    </div>
                                    <div class="venue-stat">
                                        <div class="venue-stat-value">${stats.avg_dwell_minutes || 0}m</div>
                                        <div class="venue-stat-label">Avg Dwell</div>
                                    </div>
                                    <div class="venue-stat">
                                        <div class="venue-stat-value">${Math.round((stats.return_rate || 0) * 100)}%</div>
                                        <div class="venue-stat-label">Return Rate</div>
                                    </div>
                                </div>
                                <div class="venue-actions">
                                    <a href="/analytics-dashboard/${venue.id}" class="btn">View Dashboard</a>
                                    <a href="/report/${venue.id}" class="btn btn-secondary">Report</a>
                                </div>
                            </div>
                        `;
                    }));

                    document.getElementById('venues-grid').innerHTML = venueCards.join('');
                    document.getElementById('venues-grid').style.display = 'grid';
                } catch (error) {
                    document.getElementById('loading').innerHTML = 'Error loading venues: ' + error.message;
                }
            }

            loadVenues();
        </script>
    </body>
    </html>
    """


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

    # Calculate confidence metrics from track quality data
    confidence_level = None
    visitor_range = None
    data_quality = None

    if unique > 0:
        # Get average track frames and detection confidence
        query = sqlalchemy.select(
            func.avg(events.c.track_frames).label('avg_frames'),
            func.avg(events.c.detection_conf).label('avg_conf')
        ).where(
            events.c.venue_id == venue_id,
            events.c.timestamp >= since,
            events.c.is_repeat == False  # Only primary events
        )
        quality_row = await database.fetch_one(query)

        avg_frames = quality_row["avg_frames"] if quality_row and quality_row["avg_frames"] else 5
        avg_conf = quality_row["avg_conf"] if quality_row and quality_row["avg_conf"] else 0.7

        # Confidence calculation:
        # - Track frames: more frames = higher confidence (5-30 frames maps to 0.5-1.0)
        # - Detection confidence: higher = better (0.5-1.0 maps to 0.5-1.0)
        frame_score = min(1.0, max(0.5, (avg_frames - 5) / 25 + 0.5)) if avg_frames else 0.5
        conf_score = max(0.5, min(1.0, avg_conf)) if avg_conf else 0.5
        confidence_level = round((frame_score * 0.6 + conf_score * 0.4), 2)

        # Calculate confidence interval (rough 95% CI)
        # Error margin decreases with higher confidence and more samples
        error_margin = max(1, int(unique * (1 - confidence_level) * 0.5))
        visitor_range = {
            "low": max(0, unique - error_margin),
            "high": unique + error_margin
        }

        # Data quality label
        if confidence_level >= 0.85:
            data_quality = "high"
        elif confidence_level >= 0.70:
            data_quality = "medium"
        else:
            data_quality = "low"

    return StatsResponse(
        venue_id=venue_id,
        period=f"Last {days} days",
        total_visitors=total,
        unique_visitors=unique,
        repeat_rate=round(repeats / total * 100, 1) if total > 0 else 0,
        avg_dwell_minutes=round(avg_dwell / 60, 1),
        peak_hour=peak_hour,
        gender_split=gender_split,
        age_distribution=age_dist,
        confidence_level=confidence_level,
        visitor_range=visitor_range,
        data_quality=data_quality
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


# =============================================================================
# NIELSEN-STYLE ANALYTICS ENDPOINTS (Phase 5)
# =============================================================================

@app.get("/analytics/{venue_id}/demographics")
async def get_demographics_analytics(
    venue_id: str,
    days: int = 7
):
    """
    Get demographic breakdown (Nielsen-style).
    Returns age and gender distribution with week-over-week comparison.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    prev_start = start_date - timedelta(days=days)

    # Current period
    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    current_rows = await database.fetch_all(query)

    # Previous period for comparison
    query_prev = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= prev_start,
        events.c.timestamp < start_date
    )
    prev_rows = await database.fetch_all(query_prev)

    def calc_demographics(rows):
        total = len(rows)
        if total == 0:
            return {"gender": {}, "age": {}, "total": 0}

        gender_counts = {"M": 0, "F": 0, "unknown": 0}
        age_counts = {"20s": 0, "30s": 0, "40s": 0, "50+": 0, "unknown": 0}

        for row in rows:
            g = row["gender"]
            a = row["age_bracket"]
            if g in gender_counts:
                gender_counts[g] += 1
            else:
                gender_counts["unknown"] += 1
            if a in age_counts:
                age_counts[a] += 1
            else:
                age_counts["unknown"] += 1

        return {
            "gender": {k: round(v / total * 100, 1) for k, v in gender_counts.items() if v > 0},
            "age": {k: round(v / total * 100, 1) for k, v in age_counts.items() if v > 0},
            "total": total
        }

    current = calc_demographics(current_rows)
    previous = calc_demographics(prev_rows)

    # Calculate changes
    gender_change = {}
    for k in current["gender"]:
        prev_val = previous["gender"].get(k, 0)
        gender_change[k] = round(current["gender"][k] - prev_val, 1)

    age_change = {}
    for k in current["age"]:
        prev_val = previous["age"].get(k, 0)
        age_change[k] = round(current["age"][k] - prev_val, 1)

    return {
        "venue_id": venue_id,
        "period": f"Last {days} days",
        "current": current,
        "previous": previous,
        "change": {
            "gender": gender_change,
            "age": age_change,
            "total_visitors": current["total"] - previous["total"]
        }
    }


@app.get("/analytics/{venue_id}/zones")
async def get_zone_analytics(
    venue_id: str,
    days: int = 7
):
    """
    Get zone performance analytics (Nielsen-style).
    Shows traffic, dwell time, and engagement per zone.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    rows = await database.fetch_all(query)

    # Aggregate by zone
    zones = {}
    total_visitors = set()

    for row in rows:
        zone = row["zone"] or "unknown"
        if zone not in zones:
            zones[zone] = {
                "visitors": set(),
                "total_dwell": 0,
                "event_count": 0
            }
        zones[zone]["visitors"].add(row["pseudo_id"])
        zones[zone]["total_dwell"] += row["dwell_seconds"] or 0
        zones[zone]["event_count"] += 1
        total_visitors.add(row["pseudo_id"])

    total_count = len(total_visitors) or 1  # Avoid division by zero

    result = []
    for zone, data in sorted(zones.items(), key=lambda x: len(x[1]["visitors"]), reverse=True):
        visitor_count = len(data["visitors"])
        avg_dwell = data["total_dwell"] / data["event_count"] if data["event_count"] > 0 else 0

        result.append({
            "zone": zone,
            "visitors": visitor_count,
            "traffic_percent": round(visitor_count / total_count * 100, 1),
            "avg_dwell_minutes": round(avg_dwell / 60, 1),
            "engagement": "High" if avg_dwell > 300 else "Medium" if avg_dwell > 120 else "Low"
        })

    return {
        "venue_id": venue_id,
        "period": f"Last {days} days",
        "total_unique_visitors": len(total_visitors),
        "zones": result
    }


@app.get("/analytics/{venue_id}/trends")
async def get_trend_analytics(
    venue_id: str,
    weeks: int = 8
):
    """
    Get weekly trends (Nielsen-style).
    Shows visitor counts, dwell time, and demographics over time.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(weeks=weeks)

    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    ).order_by(events.c.timestamp)

    rows = await database.fetch_all(query)

    # Group by week
    weekly = {}
    for row in rows:
        # Get ISO week
        week_start = row["timestamp"] - timedelta(days=row["timestamp"].weekday())
        week_key = week_start.strftime("%Y-%m-%d")

        if week_key not in weekly:
            weekly[week_key] = {
                "visitors": set(),
                "total_dwell": 0,
                "event_count": 0,
                "return_count": 0
            }

        weekly[week_key]["visitors"].add(row["pseudo_id"])
        weekly[week_key]["total_dwell"] += row["dwell_seconds"] or 0
        weekly[week_key]["event_count"] += 1
        if row["is_repeat"]:
            weekly[week_key]["return_count"] += 1

    # Convert to list
    result = []
    for week_key in sorted(weekly.keys()):
        data = weekly[week_key]
        visitor_count = len(data["visitors"])
        avg_dwell = data["total_dwell"] / data["event_count"] if data["event_count"] > 0 else 0
        return_rate = data["return_count"] / visitor_count * 100 if visitor_count > 0 else 0

        result.append({
            "week": week_key,
            "visitors": visitor_count,
            "avg_dwell_minutes": round(avg_dwell / 60, 1),
            "return_rate_percent": round(return_rate, 1)
        })

    # Calculate trend
    if len(result) >= 2:
        first_half = sum(r["visitors"] for r in result[:len(result)//2])
        second_half = sum(r["visitors"] for r in result[len(result)//2:])
        trend = "growing" if second_half > first_half else "declining" if second_half < first_half else "stable"
        growth_rate = round((second_half - first_half) / (first_half or 1) * 100, 1)
    else:
        trend = "insufficient_data"
        growth_rate = 0

    return {
        "venue_id": venue_id,
        "period": f"Last {weeks} weeks",
        "weekly": result,
        "trend": trend,
        "growth_rate_percent": growth_rate
    }


@app.get("/analytics/{venue_id}/summary")
async def get_executive_summary(
    venue_id: str,
    days: int = 7
):
    """
    Executive summary (Nielsen-style).
    Key metrics and insights for quick overview.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    prev_start = start_date - timedelta(days=days)

    # Current period
    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    current_rows = await database.fetch_all(query)

    # Previous period
    query_prev = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= prev_start,
        events.c.timestamp < start_date
    )
    prev_rows = await database.fetch_all(query_prev)

    def calc_metrics(rows):
        if not rows:
            return {
                "total_visitors": 0,
                "unique_visitors": 0,
                "return_visitors": 0,
                "return_rate_percent": 0,
                "avg_dwell_minutes": 0,
                "peak_hour": None,
                "peak_day": None,
                "top_demographic": None,
                "avg_engagement": None,
                "engaged_percent": None,
            }

        visitors = set()
        return_count = 0
        total_dwell = 0
        hourly = {}
        daily = {}
        demographics = {}
        engagement_scores = []
        engaged_count = 0

        for row in rows:
            visitors.add(row["pseudo_id"])
            if row["is_repeat"]:
                return_count += 1
            total_dwell += row["dwell_seconds"] or 0

            hour = row["timestamp"].hour
            day = row["timestamp"].strftime("%A")
            hourly[hour] = hourly.get(hour, 0) + 1
            daily[day] = daily.get(day, 0) + 1

            demo_key = f"{row['gender'] or '?'}/{row['age_bracket'] or '?'}"
            demographics[demo_key] = demographics.get(demo_key, 0) + 1

            # Engagement tracking
            engagement_val = row["engagement_score"]
            if engagement_val is not None:
                engagement_scores.append(engagement_val)
                if engagement_val >= 70:
                    engaged_count += 1

        unique = len(visitors)
        peak_hour = max(hourly, key=hourly.get) if hourly else None
        peak_day = max(daily, key=daily.get) if daily else None
        top_demo = max(demographics, key=demographics.get) if demographics else None

        # Calculate engagement metrics
        avg_engagement = round(sum(engagement_scores) / len(engagement_scores), 1) if engagement_scores else None
        engaged_percent = round(engaged_count / len(engagement_scores) * 100, 1) if engagement_scores else None

        return {
            "total_visitors": len(rows),
            "unique_visitors": unique,
            "return_visitors": return_count,
            "return_rate_percent": round(return_count / unique * 100, 1) if unique > 0 else 0,
            "avg_dwell_minutes": round(total_dwell / len(rows) / 60, 1) if rows else 0,
            "peak_hour": peak_hour,
            "peak_day": peak_day,
            "top_demographic": top_demo,
            "avg_engagement": avg_engagement,
            "engaged_percent": engaged_percent,
        }

    current = calc_metrics(current_rows)
    previous = calc_metrics(prev_rows)

    # Generate insights
    insights = []
    if current["unique_visitors"] > previous["unique_visitors"]:
        pct = round((current["unique_visitors"] - previous["unique_visitors"]) / (previous["unique_visitors"] or 1) * 100, 1)
        insights.append(f"Visitor traffic up {pct}% vs previous period")
    elif current["unique_visitors"] < previous["unique_visitors"]:
        pct = round((previous["unique_visitors"] - current["unique_visitors"]) / (previous["unique_visitors"] or 1) * 100, 1)
        insights.append(f"Visitor traffic down {pct}% vs previous period")

    if current["avg_dwell_minutes"] > previous["avg_dwell_minutes"]:
        insights.append(f"Dwell time improved to {current['avg_dwell_minutes']} min average")

    if current["return_rate_percent"] > 20:
        insights.append(f"Strong return rate at {current['return_rate_percent']}%")

    if current["peak_hour"]:
        insights.append(f"Peak traffic at {current['peak_hour']}:00")

    # Engagement insights
    if current["avg_engagement"] is not None:
        if current["avg_engagement"] >= 70:
            insights.append(f"High engagement score: {current['avg_engagement']}")
        elif current["avg_engagement"] >= 50:
            insights.append(f"Moderate engagement score: {current['avg_engagement']}")
        if current["engaged_percent"] and current["engaged_percent"] >= 30:
            insights.append(f"{current['engaged_percent']}% of visitors highly engaged")

    return {
        "venue_id": venue_id,
        "period": f"Last {days} days",
        "current": current,
        "previous": previous,
        "change": {
            "visitors": current["unique_visitors"] - previous["unique_visitors"],
            "visitors_percent": round((current["unique_visitors"] - previous["unique_visitors"]) / (previous["unique_visitors"] or 1) * 100, 1),
            "dwell_minutes": round(current["avg_dwell_minutes"] - previous["avg_dwell_minutes"], 1)
        },
        "insights": insights
    }


@app.get("/analytics/{venue_id}/heatmap")
async def get_hourly_heatmap(
    venue_id: str,
    weeks: int = 4
):
    """
    Get hourly heatmap data (Nielsen-style).
    Returns visitor counts by day-of-week and hour for heatmap visualization.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(weeks=weeks)

    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    rows = await database.fetch_all(query)

    # Build heatmap grid: day x hour
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap = {day: {h: 0 for h in range(24)} for day in days}

    for row in rows:
        day = row["timestamp"].strftime("%A")
        hour = row["timestamp"].hour
        if day in heatmap:
            heatmap[day][hour] += 1

    # Find max for normalization
    max_val = max(max(h.values()) for h in heatmap.values()) or 1

    # Convert to list format for frontend
    result = []
    for day in days:
        for hour in range(24):
            count = heatmap[day][hour]
            result.append({
                "day": day,
                "hour": hour,
                "count": count,
                "intensity": round(count / max_val, 2)  # 0-1 normalized
            })

    # Find peak times
    peak_times = sorted(result, key=lambda x: x["count"], reverse=True)[:5]

    return {
        "venue_id": venue_id,
        "period": f"Last {weeks} weeks",
        "heatmap": result,
        "peak_times": peak_times,
        "max_count": max_val
    }


@app.get("/analytics/{venue_id}/export")
async def export_analytics(
    venue_id: str,
    days: int = 7,
    format: str = "json"
):
    """
    Export analytics data for reporting.
    Combines all analytics into a single exportable format.
    """
    # Gather all analytics
    summary = await get_executive_summary(venue_id, days)
    demographics = await get_demographics_analytics(venue_id, days)
    zones = await get_zone_analytics(venue_id, days)
    heatmap = await get_hourly_heatmap(venue_id, weeks=max(1, days // 7))
    behavior = await get_behavior_analytics(venue_id, days)

    export_data = {
        "venue_id": venue_id,
        "generated_at": datetime.utcnow().isoformat(),
        "period_days": days,
        "executive_summary": summary,
        "demographics": demographics,
        "zone_performance": zones,
        "hourly_heatmap": {
            "peak_times": heatmap["peak_times"],
            "max_hourly_count": heatmap["max_count"]
        },
        "behavior_analytics": behavior
    }

    if format == "csv":
        # Return as downloadable CSV summary
        import io
        output = io.StringIO()
        output.write("CCTV Analytics Report\\n")
        output.write(f"Venue: {venue_id}\\n")
        output.write(f"Period: Last {days} days\\n")
        output.write(f"Generated: {datetime.utcnow().isoformat()}\\n\\n")

        output.write("KEY METRICS\\n")
        output.write(f"Unique Visitors,{summary['current']['unique_visitors']}\\n")
        output.write(f"Return Rate,{summary['current']['return_rate_percent']}%\\n")
        output.write(f"Avg Dwell Time,{summary['current']['avg_dwell_minutes']} min\\n")
        output.write(f"Peak Hour,{summary['current']['peak_hour']}:00\\n")
        if summary['current'].get('avg_engagement'):
            output.write(f"Avg Engagement Score,{summary['current']['avg_engagement']}\\n")
            output.write(f"Highly Engaged %,{summary['current']['engaged_percent']}%\\n")

        return HTMLResponse(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={venue_id}_analytics.csv"}
        )

    return export_data


# =============================================================================
# BEHAVIOR ANALYTICS ENDPOINTS (Phase 5)
# =============================================================================

@app.get("/analytics/{venue_id}/behavior")
async def get_behavior_analytics(venue_id: str, days: int = 7):
    """
    Get behavior and engagement analytics for a venue.
    Returns engagement scores, behavior type breakdown, and posture analysis.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            AVG(engagement_score) as avg_engagement,
            MIN(engagement_score) as min_engagement,
            MAX(engagement_score) as max_engagement,
            COUNT(CASE WHEN engagement_score >= 70 THEN 1 END) as highly_engaged,
            COUNT(CASE WHEN engagement_score >= 50 AND engagement_score < 70 THEN 1 END) as moderately_engaged,
            COUNT(CASE WHEN engagement_score < 50 THEN 1 END) as low_engaged,
            COUNT(engagement_score) as total_with_behavior
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND engagement_score IS NOT NULL
    """

    result = await database.fetch_one(query, {"venue_id": venue_id, "cutoff": cutoff})

    # Get behavior type breakdown
    behavior_query = """
        SELECT behavior_type, COUNT(*) as count
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND behavior_type IS NOT NULL
        GROUP BY behavior_type
        ORDER BY count DESC
    """
    behavior_rows = await database.fetch_all(behavior_query, {"venue_id": venue_id, "cutoff": cutoff})
    behavior_breakdown = {row["behavior_type"]: row["count"] for row in behavior_rows}

    # Get posture breakdown
    posture_query = """
        SELECT posture, COUNT(*) as count
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND posture IS NOT NULL
        GROUP BY posture
        ORDER BY count DESC
    """
    posture_rows = await database.fetch_all(posture_query, {"venue_id": venue_id, "cutoff": cutoff})
    posture_breakdown = {row["posture"]: row["count"] for row in posture_rows}

    # Get body orientation distribution
    orientation_query = """
        SELECT
            COUNT(CASE WHEN body_orientation > 0.3 THEN 1 END) as facing_camera,
            COUNT(CASE WHEN body_orientation <= 0.3 AND body_orientation >= -0.3 THEN 1 END) as sideways,
            COUNT(CASE WHEN body_orientation < -0.3 THEN 1 END) as facing_away,
            AVG(body_orientation) as avg_orientation
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND body_orientation IS NOT NULL
    """
    orientation = await database.fetch_one(orientation_query, {"venue_id": venue_id, "cutoff": cutoff})

    total = result["total_with_behavior"] or 0

    return {
        "venue_id": venue_id,
        "period_days": days,
        "engagement": {
            "average_score": round(result["avg_engagement"] or 0, 1),
            "min_score": round(result["min_engagement"] or 0, 1),
            "max_score": round(result["max_engagement"] or 0, 1),
            "highly_engaged_count": result["highly_engaged"] or 0,
            "highly_engaged_percent": round((result["highly_engaged"] or 0) / max(total, 1) * 100, 1),
            "moderately_engaged_count": result["moderately_engaged"] or 0,
            "low_engaged_count": result["low_engaged"] or 0,
        },
        "behavior_types": behavior_breakdown,
        "postures": posture_breakdown,
        "body_orientation": {
            "facing_camera": orientation["facing_camera"] or 0,
            "sideways": orientation["sideways"] or 0,
            "facing_away": orientation["facing_away"] or 0,
            "avg_orientation": round(orientation["avg_orientation"] or 0, 2),
        },
        "total_analyzed": total,
    }


@app.get("/analytics/{venue_id}/behavior/hourly")
async def get_behavior_hourly(venue_id: str, days: int = 7):
    """
    Get hourly engagement patterns.
    Shows when visitors are most engaged throughout the day.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            CAST(strftime('%%H', timestamp) AS INTEGER) as hour,
            AVG(engagement_score) as avg_engagement,
            COUNT(*) as visitor_count,
            SUM(CASE WHEN behavior_type = 'engaged' THEN 1 ELSE 0 END) as engaged_count,
            SUM(CASE WHEN behavior_type = 'browsing' THEN 1 ELSE 0 END) as browsing_count,
            SUM(CASE WHEN behavior_type = 'passing' THEN 1 ELSE 0 END) as passing_count
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND engagement_score IS NOT NULL
        GROUP BY hour
        ORDER BY hour
    """

    rows = await database.fetch_all(query, {"venue_id": venue_id, "cutoff": cutoff})

    hourly_data = []
    peak_engagement_hour = 0
    peak_engagement_score = 0

    for row in rows:
        avg_eng = row["avg_engagement"] or 0
        if avg_eng > peak_engagement_score:
            peak_engagement_score = avg_eng
            peak_engagement_hour = row["hour"]

        hourly_data.append({
            "hour": row["hour"],
            "avg_engagement": round(avg_eng, 1),
            "visitor_count": row["visitor_count"],
            "engaged_count": row["engaged_count"] or 0,
            "browsing_count": row["browsing_count"] or 0,
            "passing_count": row["passing_count"] or 0,
        })

    return {
        "venue_id": venue_id,
        "period_days": days,
        "hourly_engagement": hourly_data,
        "peak_engagement_hour": peak_engagement_hour,
        "peak_engagement_score": round(peak_engagement_score, 1),
        "insight": f"Visitors are most engaged at {peak_engagement_hour}:00 (avg score: {round(peak_engagement_score, 1)})"
    }


@app.get("/analytics/{venue_id}/behavior/zones")
async def get_behavior_by_zone(venue_id: str, days: int = 7):
    """
    Get engagement metrics by zone.
    Shows which areas of the venue have the most engaged visitors.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            zone,
            AVG(engagement_score) as avg_engagement,
            COUNT(*) as visitor_count,
            AVG(dwell_seconds) as avg_dwell,
            SUM(CASE WHEN behavior_type = 'engaged' THEN 1 ELSE 0 END) as engaged_count,
            SUM(CASE WHEN behavior_type = 'browsing' THEN 1 ELSE 0 END) as browsing_count,
            SUM(CASE WHEN behavior_type = 'waiting' THEN 1 ELSE 0 END) as waiting_count,
            SUM(CASE WHEN behavior_type = 'passing' THEN 1 ELSE 0 END) as passing_count
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND engagement_score IS NOT NULL
        GROUP BY zone
        ORDER BY avg_engagement DESC
    """

    rows = await database.fetch_all(query, {"venue_id": venue_id, "cutoff": cutoff})

    zones = []
    for row in rows:
        total = row["visitor_count"] or 1
        zones.append({
            "zone": row["zone"],
            "avg_engagement": round(row["avg_engagement"] or 0, 1),
            "visitor_count": row["visitor_count"],
            "avg_dwell_seconds": round(row["avg_dwell"] or 0, 1),
            "behavior_mix": {
                "engaged": round((row["engaged_count"] or 0) / total * 100, 1),
                "browsing": round((row["browsing_count"] or 0) / total * 100, 1),
                "waiting": round((row["waiting_count"] or 0) / total * 100, 1),
                "passing": round((row["passing_count"] or 0) / total * 100, 1),
            }
        })

    # Find best and worst zones
    best_zone = zones[0]["zone"] if zones else "unknown"
    worst_zone = zones[-1]["zone"] if zones else "unknown"

    return {
        "venue_id": venue_id,
        "period_days": days,
        "zones": zones,
        "insights": {
            "highest_engagement_zone": best_zone,
            "lowest_engagement_zone": worst_zone,
            "recommendation": f"Focus on improving engagement in the '{worst_zone}' zone" if worst_zone != best_zone else "All zones performing similarly"
        }
    }


@app.get("/report/{venue_id}", response_class=HTMLResponse)
async def generate_report(venue_id: str, days: int = 7):
    """
    Generate printable HTML report (Nielsen-style).
    Open in browser and print to PDF.
    """
    # Gather data
    summary = await get_executive_summary(venue_id, days)
    demographics = await get_demographics_analytics(venue_id, days)
    zones = await get_zone_analytics(venue_id, days)

    current = summary["current"]
    change = summary["change"]

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics Report - {venue_id}</title>
        <style>
            @media print {{
                body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
                .no-print {{ display: none; }}
            }}
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #fff;
                color: #1a1a1a;
                padding: 40px;
                max-width: 900px;
                margin: 0 auto;
            }}
            .header {{
                border-bottom: 3px solid #3b82f6;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{ font-size: 28px; margin-bottom: 5px; }}
            .header .subtitle {{ color: #666; }}
            .header .period {{ color: #3b82f6; font-weight: 600; margin-top: 10px; }}

            .section {{ margin-bottom: 40px; }}
            .section h2 {{
                font-size: 18px;
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}

            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
            }}
            .kpi-card {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
            }}
            .kpi-value {{ font-size: 32px; font-weight: bold; color: #1a1a1a; }}
            .kpi-label {{ color: #666; font-size: 12px; margin-top: 5px; text-transform: uppercase; }}
            .kpi-change {{ font-size: 12px; margin-top: 5px; }}
            .kpi-change.positive {{ color: #22c55e; }}
            .kpi-change.negative {{ color: #ef4444; }}

            .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}

            .demo-bars {{ }}
            .demo-bar {{
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }}
            .demo-label {{ width: 80px; font-size: 14px; color: #666; }}
            .demo-track {{
                flex: 1;
                height: 24px;
                background: #e5e7eb;
                border-radius: 4px;
                overflow: hidden;
            }}
            .demo-fill {{
                height: 100%;
                background: #3b82f6;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 8px;
                color: white;
                font-size: 12px;
                font-weight: 600;
            }}
            .demo-fill.female {{ background: #ec4899; }}

            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e5e7eb;
            }}
            th {{ background: #f8f9fa; font-weight: 600; color: #666; font-size: 12px; text-transform: uppercase; }}

            .insights {{
                background: #eff6ff;
                border-left: 4px solid #3b82f6;
                padding: 20px;
                border-radius: 0 8px 8px 0;
            }}
            .insights h3 {{ margin-bottom: 15px; color: #1e40af; }}
            .insights ul {{ list-style: none; }}
            .insights li {{ padding: 8px 0; padding-left: 20px; position: relative; }}
            .insights li:before {{ content: "→"; position: absolute; left: 0; color: #3b82f6; }}

            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #888;
                font-size: 12px;
                text-align: center;
            }}

            .print-btn {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: #3b82f6;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
            }}
            .print-btn:hover {{ background: #2563eb; }}
        </style>
    </head>
    <body>
        <button class="print-btn no-print" onclick="window.print()">Print / Save PDF</button>

        <div class="header">
            <h1>📊 Analytics Report</h1>
            <div class="subtitle">Venue: {venue_id}</div>
            <div class="period">Period: Last {days} days • Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
        </div>

        <div class="section">
            <h2>Key Performance Indicators</h2>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value">{current['unique_visitors']}</div>
                    <div class="kpi-label">Unique Visitors</div>
                    <div class="kpi-change {'positive' if change['visitors'] >= 0 else 'negative'}">
                        {'+' if change['visitors'] >= 0 else ''}{change['visitors']} ({'+' if change['visitors_percent'] >= 0 else ''}{change['visitors_percent']}%)
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{current['return_rate_percent']}%</div>
                    <div class="kpi-label">Return Rate</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{current['avg_dwell_minutes']}</div>
                    <div class="kpi-label">Avg Dwell (min)</div>
                    <div class="kpi-change {'positive' if change['dwell_minutes'] >= 0 else 'negative'}">
                        {'+' if change['dwell_minutes'] >= 0 else ''}{change['dwell_minutes']} min
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{current['peak_hour'] or '-'}:00</div>
                    <div class="kpi-label">Peak Hour</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Demographics</h2>
            <div class="two-col">
                <div>
                    <h4 style="margin-bottom: 15px; color: #666;">Gender Distribution</h4>
                    <div class="demo-bars">
                        <div class="demo-bar">
                            <span class="demo-label">Male</span>
                            <div class="demo-track">
                                <div class="demo-fill" style="width: {demographics['current']['gender'].get('M', 0)}%">
                                    {demographics['current']['gender'].get('M', 0)}%
                                </div>
                            </div>
                        </div>
                        <div class="demo-bar">
                            <span class="demo-label">Female</span>
                            <div class="demo-track">
                                <div class="demo-fill female" style="width: {demographics['current']['gender'].get('F', 0)}%">
                                    {demographics['current']['gender'].get('F', 0)}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div>
                    <h4 style="margin-bottom: 15px; color: #666;">Age Distribution</h4>
                    <div class="demo-bars">
                        {''.join(f'''
                        <div class="demo-bar">
                            <span class="demo-label">{age}</span>
                            <div class="demo-track">
                                <div class="demo-fill" style="width: {pct}%; background: {'#3b82f6' if age == '20s' else '#8b5cf6' if age == '30s' else '#f59e0b' if age == '40s' else '#ef4444'}">
                                    {pct}%
                                </div>
                            </div>
                        </div>
                        ''' for age, pct in demographics['current']['age'].items())}
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Zone Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Zone</th>
                        <th>Visitors</th>
                        <th>Traffic %</th>
                        <th>Avg Dwell</th>
                        <th>Engagement</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td>{z['zone']}</td>
                        <td>{z['visitors']}</td>
                        <td>{z['traffic_percent']}%</td>
                        <td>{z['avg_dwell_minutes']} min</td>
                        <td>{z['engagement']}</td>
                    </tr>
                    ''' for z in zones['zones'][:10])}
                </tbody>
            </table>
        </div>

        <div class="section">
            <div class="insights">
                <h3>Key Insights</h3>
                <ul>
                    {''.join(f'<li>{insight}</li>' for insight in summary['insights']) if summary['insights'] else '<li>Process more videos to generate insights</li>'}
                </ul>
            </div>
        </div>

        <div class="footer">
            Generated by CCTV Analytics Platform • {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
        </div>
    </body>
    </html>
    """


# =============================================================================
# ALERTS & ANOMALY DETECTION
# =============================================================================

async def check_anomalies(venue_id: str) -> List[Dict]:
    """
    Check for anomalies and generate alerts.
    Called after video processing completes.
    """
    detected_alerts = []

    # Get recent data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    baseline_start = start_date - timedelta(days=7)

    # Current week events
    query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    current_rows = await database.fetch_all(query)

    # Previous week events (baseline)
    query_baseline = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= baseline_start,
        events.c.timestamp < start_date
    )
    baseline_rows = await database.fetch_all(query_baseline)

    if not baseline_rows:
        return []  # Not enough data for comparison

    current_visitors = len(set(r["pseudo_id"] for r in current_rows))
    baseline_visitors = len(set(r["pseudo_id"] for r in baseline_rows))

    # Check for significant traffic changes
    if baseline_visitors > 0:
        change_pct = (current_visitors - baseline_visitors) / baseline_visitors * 100

        if change_pct > 50:
            detected_alerts.append({
                "alert_type": "traffic_spike",
                "severity": "info",
                "title": "Traffic Spike Detected",
                "message": f"Visitor traffic up {change_pct:.0f}% vs previous week ({current_visitors} vs {baseline_visitors})",
                "data": {"current": current_visitors, "baseline": baseline_visitors, "change_pct": change_pct}
            })
        elif change_pct < -30:
            detected_alerts.append({
                "alert_type": "traffic_drop",
                "severity": "warning",
                "title": "Traffic Drop Detected",
                "message": f"Visitor traffic down {abs(change_pct):.0f}% vs previous week ({current_visitors} vs {baseline_visitors})",
                "data": {"current": current_visitors, "baseline": baseline_visitors, "change_pct": change_pct}
            })

    # Check for unusual hour activity (between midnight and 5am)
    unusual_hours = [r for r in current_rows if r["timestamp"].hour < 5]
    if len(unusual_hours) > 10:
        detected_alerts.append({
            "alert_type": "unusual_hours",
            "severity": "info",
            "title": "After-Hours Activity",
            "message": f"Detected {len(unusual_hours)} events between midnight and 5am",
            "data": {"count": len(unusual_hours)}
        })

    # Store alerts
    for alert_data in detected_alerts:
        insert_query = alerts.insert().values(
            venue_id=venue_id,
            alert_type=alert_data["alert_type"],
            severity=alert_data["severity"],
            title=alert_data["title"],
            message=alert_data["message"],
            data=alert_data["data"],
            created_at=datetime.utcnow()
        )
        await database.execute(insert_query)

    return detected_alerts


@app.get("/api/alerts")
async def list_alerts(
    venue_id: Optional[str] = None,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = 50
):
    """List alerts, optionally filtered."""
    query = sqlalchemy.select(alerts).order_by(alerts.c.created_at.desc()).limit(limit)

    if venue_id:
        query = query.where(alerts.c.venue_id == venue_id)
    if severity:
        query = query.where(alerts.c.severity == severity)
    if acknowledged is not None:
        query = query.where(alerts.c.acknowledged == acknowledged)

    rows = await database.fetch_all(query)

    return {
        "alerts": [
            {
                "id": r["id"],
                "venue_id": r["venue_id"],
                "alert_type": r["alert_type"],
                "severity": r["severity"],
                "title": r["title"],
                "message": r["message"],
                "data": r["data"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "acknowledged": r["acknowledged"]
            }
            for r in rows
        ]
    }


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """Mark an alert as acknowledged."""
    query = alerts.update().where(alerts.c.id == alert_id).values(
        acknowledged=True,
        acknowledged_at=datetime.utcnow()
    )
    result = await database.execute(query)

    if result == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"message": "Alert acknowledged", "alert_id": alert_id}


@app.post("/api/alerts/check/{venue_id}")
async def trigger_anomaly_check(venue_id: str):
    """Manually trigger anomaly detection for a venue."""
    detected = await check_anomalies(venue_id)
    return {
        "venue_id": venue_id,
        "alerts_generated": len(detected),
        "alerts": detected
    }


# =============================================================================
# BENCHMARKING & COMPARISON
# =============================================================================

@app.get("/api/benchmark/venues")
async def compare_venues(venue_ids: str, days: int = 7):
    """
    Compare multiple venues side by side.
    Pass venue IDs as comma-separated string: ?venue_ids=venue1,venue2,venue3
    """
    ids = [v.strip() for v in venue_ids.split(",") if v.strip()]

    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 venue IDs to compare")
    if len(ids) > 10:
        raise HTTPException(status_code=400, detail="Max 10 venues for comparison")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    results = []
    for venue_id in ids:
        query = sqlalchemy.select(events).where(
            events.c.venue_id == venue_id,
            events.c.timestamp >= start_date,
            events.c.timestamp < end_date
        )
        rows = await database.fetch_all(query)

        if not rows:
            results.append({
                "venue_id": venue_id,
                "unique_visitors": 0,
                "avg_dwell_minutes": 0,
                "return_rate_percent": 0
            })
            continue

        visitors = set()
        return_count = 0
        total_dwell = 0

        for row in rows:
            visitors.add(row["pseudo_id"])
            if row["is_repeat"]:
                return_count += 1
            total_dwell += row["dwell_seconds"] or 0

        unique = len(visitors)
        results.append({
            "venue_id": venue_id,
            "unique_visitors": unique,
            "avg_dwell_minutes": round(total_dwell / len(rows) / 60, 1) if rows else 0,
            "return_rate_percent": round(return_count / unique * 100, 1) if unique > 0 else 0
        })

    # Sort by visitors descending
    results.sort(key=lambda x: x["unique_visitors"], reverse=True)

    # Calculate averages for "industry" comparison
    avg_visitors = sum(r["unique_visitors"] for r in results) / len(results) if results else 0
    avg_dwell = sum(r["avg_dwell_minutes"] for r in results) / len(results) if results else 0
    avg_return = sum(r["return_rate_percent"] for r in results) / len(results) if results else 0

    return {
        "period": f"Last {days} days",
        "venues": results,
        "averages": {
            "unique_visitors": round(avg_visitors, 1),
            "avg_dwell_minutes": round(avg_dwell, 1),
            "return_rate_percent": round(avg_return, 1)
        }
    }


@app.get("/api/benchmark/industry")
async def get_industry_benchmarks(venue_type: str = "bar", days: int = 7):
    """
    Get industry benchmarks based on venue type.
    Calculates averages across all venues of the same type.
    """
    # Get all venues of this type
    venue_query = sqlalchemy.select(venues.c.id).where(venues.c.venue_type == venue_type)
    venue_rows = await database.fetch_all(venue_query)
    venue_ids = [r["id"] for r in venue_rows]

    if not venue_ids:
        # Return generic benchmarks if no venues of this type
        return {
            "venue_type": venue_type,
            "sample_size": 0,
            "benchmarks": {
                "avg_daily_visitors": 50,
                "avg_dwell_minutes": 25,
                "return_rate_percent": 20,
                "peak_hour": 20
            },
            "note": "Generic industry estimates - no venue data available"
        }

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Aggregate across all venues
    all_visitors = []
    all_dwells = []
    all_returns = []
    hourly_counts = {}

    for venue_id in venue_ids:
        query = sqlalchemy.select(events).where(
            events.c.venue_id == venue_id,
            events.c.timestamp >= start_date,
            events.c.timestamp < end_date
        )
        rows = await database.fetch_all(query)

        if rows:
            visitors = len(set(r["pseudo_id"] for r in rows))
            all_visitors.append(visitors / days)  # Daily average

            total_dwell = sum(r["dwell_seconds"] or 0 for r in rows)
            all_dwells.append(total_dwell / len(rows) / 60 if rows else 0)

            return_count = sum(1 for r in rows if r["is_repeat"])
            all_returns.append(return_count / visitors * 100 if visitors > 0 else 0)

            for row in rows:
                h = row["timestamp"].hour
                hourly_counts[h] = hourly_counts.get(h, 0) + 1

    peak_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else 20

    return {
        "venue_type": venue_type,
        "sample_size": len(venue_ids),
        "period": f"Last {days} days",
        "benchmarks": {
            "avg_daily_visitors": round(sum(all_visitors) / len(all_visitors), 1) if all_visitors else 0,
            "avg_dwell_minutes": round(sum(all_dwells) / len(all_dwells), 1) if all_dwells else 0,
            "return_rate_percent": round(sum(all_returns) / len(all_returns), 1) if all_returns else 0,
            "peak_hour": peak_hour
        }
    }


# =============================================================================
# RETURN VISITOR ANALYTICS (ReID)
# =============================================================================

@app.get("/api/visitors/{venue_id}")
async def get_known_visitors(
    venue_id: str,
    limit: int = 100,
    sort_by: str = "last_seen"  # last_seen, first_seen, visit_count
):
    """
    Get all known visitors for a venue.
    These are visitors with face embeddings that can be tracked across sessions.
    """
    # Build sort order
    if sort_by == "first_seen":
        order = visitor_embeddings.c.first_seen.desc()
    elif sort_by == "visit_count":
        order = visitor_embeddings.c.visit_count.desc()
    else:
        order = visitor_embeddings.c.last_seen.desc()

    query = sqlalchemy.select(
        visitor_embeddings.c.visitor_id,
        visitor_embeddings.c.first_seen,
        visitor_embeddings.c.last_seen,
        visitor_embeddings.c.visit_count,
        visitor_embeddings.c.total_dwell_seconds,
        visitor_embeddings.c.age_bracket,
        visitor_embeddings.c.gender,
        visitor_embeddings.c.quality_score
    ).where(
        visitor_embeddings.c.venue_id == venue_id
    ).order_by(order).limit(limit)

    rows = await database.fetch_all(query)

    visitors = []
    for row in rows:
        visitors.append({
            "visitor_id": row["visitor_id"],
            "first_seen": row["first_seen"].isoformat() if row["first_seen"] else None,
            "last_seen": row["last_seen"].isoformat() if row["last_seen"] else None,
            "visit_count": row["visit_count"],
            "total_dwell_minutes": round((row["total_dwell_seconds"] or 0) / 60, 1),
            "avg_dwell_minutes": round((row["total_dwell_seconds"] or 0) / max(row["visit_count"], 1) / 60, 1),
            "age_bracket": row["age_bracket"],
            "gender": row["gender"],
            "loyalty_score": calculate_loyalty_score(row)
        })

    return {
        "venue_id": venue_id,
        "total_known_visitors": len(visitors),
        "visitors": visitors
    }


def calculate_loyalty_score(visitor_row) -> str:
    """Calculate a loyalty tier based on visit count and recency."""
    visits = visitor_row["visit_count"] or 1
    last_seen = visitor_row["last_seen"]

    if last_seen:
        days_since = (datetime.utcnow() - last_seen).days
    else:
        days_since = 999

    # Loyalty tiers
    if visits >= 10 and days_since <= 30:
        return "VIP"
    elif visits >= 5 and days_since <= 30:
        return "Regular"
    elif visits >= 2 and days_since <= 60:
        return "Returning"
    elif days_since > 90:
        return "Lapsed"
    else:
        return "New"


@app.get("/api/visitors/{venue_id}/stats")
async def get_visitor_loyalty_stats(venue_id: str):
    """
    Get loyalty statistics for a venue.
    Shows distribution of visitor loyalty tiers.
    """
    query = sqlalchemy.select(visitor_embeddings).where(
        visitor_embeddings.c.venue_id == venue_id
    )
    rows = await database.fetch_all(query)

    if not rows:
        return {
            "venue_id": venue_id,
            "total_known_visitors": 0,
            "loyalty_distribution": {},
            "avg_visits_per_visitor": 0,
            "avg_lifetime_dwell_minutes": 0
        }

    # Calculate stats
    loyalty_counts = {"VIP": 0, "Regular": 0, "Returning": 0, "New": 0, "Lapsed": 0}
    total_visits = 0
    total_dwell = 0

    for row in rows:
        tier = calculate_loyalty_score(row)
        loyalty_counts[tier] += 1
        total_visits += row["visit_count"] or 1
        total_dwell += row["total_dwell_seconds"] or 0

    total_visitors = len(rows)

    return {
        "venue_id": venue_id,
        "total_known_visitors": total_visitors,
        "loyalty_distribution": loyalty_counts,
        "loyalty_percentages": {
            k: round(v / total_visitors * 100, 1) for k, v in loyalty_counts.items() if v > 0
        },
        "avg_visits_per_visitor": round(total_visits / total_visitors, 1),
        "avg_lifetime_dwell_minutes": round(total_dwell / total_visitors / 60, 1),
        "total_return_visits": total_visits - total_visitors  # Total visits minus first visits
    }


@app.get("/api/visitors/{venue_id}/history/{visitor_id}")
async def get_visitor_history(venue_id: str, visitor_id: str):
    """
    Get visit history for a specific visitor.
    Shows all sessions and events for this person.
    """
    # Get visitor info
    visitor_query = sqlalchemy.select(visitor_embeddings).where(
        visitor_embeddings.c.venue_id == venue_id,
        visitor_embeddings.c.visitor_id == visitor_id
    )
    visitor = await database.fetch_one(visitor_query)

    if not visitor:
        raise HTTPException(status_code=404, detail="Visitor not found")

    # Get all events for this visitor
    events_query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.pseudo_id == visitor_id
    ).order_by(events.c.timestamp.desc()).limit(100)

    event_rows = await database.fetch_all(events_query)

    # Group events by date (session)
    sessions = {}
    for row in event_rows:
        date_key = row["timestamp"].strftime("%Y-%m-%d")
        if date_key not in sessions:
            sessions[date_key] = {
                "date": date_key,
                "events": [],
                "total_dwell": 0,
                "zones": set()
            }
        sessions[date_key]["events"].append({
            "timestamp": row["timestamp"].isoformat(),
            "zone": row["zone"],
            "dwell_seconds": row["dwell_seconds"]
        })
        sessions[date_key]["total_dwell"] += row["dwell_seconds"] or 0
        sessions[date_key]["zones"].add(row["zone"])

    # Convert to list
    session_list = []
    for date_key in sorted(sessions.keys(), reverse=True):
        s = sessions[date_key]
        session_list.append({
            "date": s["date"],
            "event_count": len(s["events"]),
            "total_dwell_minutes": round(s["total_dwell"] / 60, 1),
            "zones_visited": list(s["zones"])
        })

    return {
        "venue_id": venue_id,
        "visitor_id": visitor_id,
        "profile": {
            "first_seen": visitor["first_seen"].isoformat() if visitor["first_seen"] else None,
            "last_seen": visitor["last_seen"].isoformat() if visitor["last_seen"] else None,
            "visit_count": visitor["visit_count"],
            "total_dwell_minutes": round((visitor["total_dwell_seconds"] or 0) / 60, 1),
            "age_bracket": visitor["age_bracket"],
            "gender": visitor["gender"],
            "loyalty_tier": calculate_loyalty_score(visitor)
        },
        "sessions": session_list
    }


@app.get("/api/visitors/{venue_id}/returning")
async def get_returning_visitor_analytics(
    venue_id: str,
    days: int = 30
):
    """
    Get analytics specifically about returning visitors.
    Useful for measuring loyalty and retention.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Get all events in period
    events_query = sqlalchemy.select(events).where(
        events.c.venue_id == venue_id,
        events.c.timestamp >= start_date,
        events.c.timestamp < end_date
    )
    event_rows = await database.fetch_all(events_query)

    # Get known visitors
    visitors_query = sqlalchemy.select(visitor_embeddings).where(
        visitor_embeddings.c.venue_id == venue_id
    )
    visitor_rows = await database.fetch_all(visitors_query)

    # Create lookup of known visitors
    known_visitor_ids = {r["visitor_id"] for r in visitor_rows}
    known_visitors_map = {r["visitor_id"]: r for r in visitor_rows}

    # Analyze events
    total_events = len(event_rows)
    unique_visitors = set()
    known_visitors_seen = set()
    return_events = 0

    for row in event_rows:
        unique_visitors.add(row["pseudo_id"])
        if row["pseudo_id"] in known_visitor_ids:
            known_visitors_seen.add(row["pseudo_id"])
        if row["is_repeat"]:
            return_events += 1

    # Calculate retention (visitors who came back)
    returning_visitors = [
        v for v in visitor_rows
        if (v["visit_count"] or 1) > 1 and v["visitor_id"] in known_visitors_seen
    ]

    return {
        "venue_id": venue_id,
        "period": f"Last {days} days",
        "summary": {
            "total_visitors": len(unique_visitors),
            "known_visitors": len(known_visitors_seen),
            "tracking_rate": round(len(known_visitors_seen) / len(unique_visitors) * 100, 1) if unique_visitors else 0,
            "return_events": return_events,
            "return_event_rate": round(return_events / total_events * 100, 1) if total_events > 0 else 0
        },
        "retention": {
            "visitors_with_multiple_visits": len(returning_visitors),
            "retention_rate": round(len(returning_visitors) / len(known_visitors_seen) * 100, 1) if known_visitors_seen else 0
        },
        "top_returning_visitors": [
            {
                "visitor_id": v["visitor_id"],
                "visit_count": v["visit_count"],
                "last_seen": v["last_seen"].isoformat() if v["last_seen"] else None,
                "loyalty_tier": calculate_loyalty_score(v)
            }
            for v in sorted(returning_visitors, key=lambda x: x["visit_count"] or 0, reverse=True)[:10]
        ]
    }


@app.get("/health")
async def health():
    """Health check for Railway."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# =============================================================================
# VIDEO PROCESSING ENDPOINTS
# =============================================================================

@app.get("/process", response_class=HTMLResponse)
async def process_page():
    """Video processing page - upload or paste YouTube URL."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Process Video - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }
            /* Navigation */
            nav {
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            nav .nav-inner {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }
            nav .logo {
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }
            nav .logo span { color: #3b82f6; }
            nav .nav-links { display: flex; gap: 30px; }
            nav .nav-links a {
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }
            nav .nav-links a:hover { color: #fff; }
            nav .nav-links a.active { color: #3b82f6; }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 40px 20px;
            }
            h1 { color: #fff; margin-bottom: 10px; }
            .subtitle { color: #888; margin-bottom: 40px; }
            .card {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 30px;
                margin-bottom: 20px;
            }
            .card h3 { margin-top: 0; color: #fff; }
            label { display: block; margin-bottom: 8px; color: #aaa; font-size: 14px; }
            input[type="text"], input[type="file"] {
                width: 100%;
                padding: 12px 16px;
                border: 1px solid #333;
                border-radius: 8px;
                background: #0a0a0a;
                color: #fff;
                font-size: 16px;
                margin-bottom: 20px;
            }
            input[type="text"]:focus { border-color: #0066ff; outline: none; }
            input[type="file"] { padding: 10px; }
            button {
                background: #0066ff;
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: background 0.2s;
            }
            button:hover { background: #0052cc; }
            button:disabled { background: #333; cursor: not-allowed; }
            .divider {
                text-align: center;
                margin: 30px 0;
                color: #555;
                position: relative;
            }
            .divider::before, .divider::after {
                content: '';
                position: absolute;
                top: 50%;
                width: 45%;
                height: 1px;
                background: #333;
            }
            .divider::before { left: 0; }
            .divider::after { right: 0; }
            #progress-container {
                display: none;
                margin-top: 30px;
            }
            .progress-bar {
                background: #333;
                border-radius: 8px;
                height: 20px;
                overflow: hidden;
                margin-bottom: 15px;
            }
            .progress-fill {
                background: linear-gradient(90deg, #0066ff, #00ccff);
                height: 100%;
                width: 0%;
                transition: width 0.3s;
            }
            #status-message {
                color: #888;
                font-size: 14px;
                margin-bottom: 20px;
            }
            #result {
                display: none;
                background: #0d2818;
                border: 1px solid #1e5631;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
            }
            #result.error {
                background: #2d1010;
                border-color: #5c1e1e;
            }
            #result h4 { margin: 0 0 10px 0; color: #4ade80; }
            #result.error h4 { color: #f87171; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-top: 15px;
            }
            .stat-box {
                background: #0a0a0a;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value { font-size: 24px; font-weight: bold; color: #fff; }
            .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
            .btn-secondary {
                background: #333;
                margin-top: 20px;
            }
            .btn-secondary:hover { background: #444; }
            a { color: #3b82f6; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/process" class="active">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container">
        <h1>🎥 Process Video</h1>
        <p class="subtitle">Upload a video file or paste a YouTube URL to analyze visitor traffic</p>

        <div class="card">
            <h3>Option 1: YouTube URL</h3>
            <label for="youtube-url">Paste YouTube URL (bar, restaurant, retail scenes work best)</label>
            <input type="text" id="youtube-url" placeholder="https://www.youtube.com/watch?v=...">
            <button onclick="processYouTube()" id="btn-youtube">Process YouTube Video</button>
        </div>

        <div class="divider">OR</div>

        <div class="card">
            <h3>Option 2: Upload Video</h3>
            <label for="video-file">Select video file (MP4, MOV, AVI)</label>
            <input type="file" id="video-file" accept="video/*">
            <button onclick="processUpload()" id="btn-upload">Process Uploaded Video</button>
        </div>

        <div class="card">
            <h3>Venue Details</h3>
            <label for="venue-id">Venue ID</label>
            <input type="text" id="venue-id" value="demo_venue" placeholder="my_bar">

            <label for="venue-name">Venue Name</label>
            <input type="text" id="venue-name" placeholder="My Restaurant">

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div>
                    <label for="venue-lat">Latitude</label>
                    <input type="number" id="venue-lat" step="0.0001" placeholder="-26.2041">
                </div>
                <div>
                    <label for="venue-lng">Longitude</label>
                    <input type="number" id="venue-lng" step="0.0001" placeholder="28.0473">
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div>
                    <label for="venue-city">City</label>
                    <input type="text" id="venue-city" placeholder="Johannesburg">
                </div>
                <div>
                    <label for="venue-country">Country</label>
                    <input type="text" id="venue-country" placeholder="South Africa">
                </div>
            </div>

            <label for="venue-type">Venue Type</label>
            <select id="venue-type" style="width: 100%; padding: 12px; background: #0a0a0a; color: #fff; border: 1px solid #333; border-radius: 8px; margin-bottom: 20px;">
                <option value="">Select type...</option>
                <option value="bar">Bar</option>
                <option value="restaurant">Restaurant</option>
                <option value="cafe">Cafe</option>
                <option value="retail">Retail Store</option>
                <option value="nightclub">Nightclub</option>
                <option value="hotel">Hotel/Lodge</option>
                <option value="mall">Shopping Mall</option>
                <option value="other">Other</option>
            </select>

            <div id="map-container" style="height: 200px; border-radius: 8px; margin-bottom: 15px; background: #1a1a1a; display: flex; align-items: center; justify-content: center; color: #666;">
                <div id="map" style="width: 100%; height: 100%; border-radius: 8px;"></div>
            </div>
            <p style="font-size: 12px; color: #666; margin: 0;">Click map to set location, or enter coordinates manually</p>
        </div>

        <!-- Leaflet CSS/JS for map -->
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

        <div id="progress-container" class="card">
            <h3>Processing...</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            <div id="status-message">Starting...</div>
        </div>

        <div id="result">
            <h4 id="result-title">Processing Complete</h4>
            <p id="result-message"></p>
            <div class="stats-grid" id="stats-grid">
                <div class="stat-box">
                    <div class="stat-value" id="stat-frames">0</div>
                    <div class="stat-label">Frames Processed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="stat-visitors">0</div>
                    <div class="stat-label">Unique Visitors</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="stat-events">0</div>
                    <div class="stat-label">Events Created</div>
                </div>
            </div>
            <button class="btn-secondary" onclick="viewDashboard()">View Dashboard &rarr;</button>
        </div>

        <script>
            let currentJobId = null;
            let statusInterval = null;
            let venueId = 'demo_venue';
            let map = null;
            let marker = null;

            // Initialize map centered on Africa
            document.addEventListener('DOMContentLoaded', function() {
                map = L.map('map').setView([-26.2041, 28.0473], 4);  // Centered on South Africa
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; OpenStreetMap contributors'
                }).addTo(map);

                // Click to set location
                map.on('click', function(e) {
                    setLocation(e.latlng.lat, e.latlng.lng);
                });

                // Update map when inputs change
                document.getElementById('venue-lat').addEventListener('change', updateMapFromInputs);
                document.getElementById('venue-lng').addEventListener('change', updateMapFromInputs);
            });

            function setLocation(lat, lng) {
                document.getElementById('venue-lat').value = lat.toFixed(6);
                document.getElementById('venue-lng').value = lng.toFixed(6);

                if (marker) {
                    marker.setLatLng([lat, lng]);
                } else {
                    marker = L.marker([lat, lng]).addTo(map);
                }
                map.setView([lat, lng], 12);
            }

            function updateMapFromInputs() {
                const lat = parseFloat(document.getElementById('venue-lat').value);
                const lng = parseFloat(document.getElementById('venue-lng').value);
                if (!isNaN(lat) && !isNaN(lng)) {
                    setLocation(lat, lng);
                }
            }

            function getVenueData() {
                return {
                    venue_id: document.getElementById('venue-id').value.trim() || 'demo_venue',
                    venue_name: document.getElementById('venue-name').value.trim(),
                    latitude: parseFloat(document.getElementById('venue-lat').value) || null,
                    longitude: parseFloat(document.getElementById('venue-lng').value) || null,
                    city: document.getElementById('venue-city').value.trim(),
                    country: document.getElementById('venue-country').value.trim(),
                    venue_type: document.getElementById('venue-type').value
                };
            }

            function setButtonsEnabled(enabled) {
                document.getElementById('btn-youtube').disabled = !enabled;
                document.getElementById('btn-upload').disabled = !enabled;
            }

            function showProgress() {
                document.getElementById('progress-container').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                setButtonsEnabled(false);
            }

            function updateProgress(percent, message) {
                document.getElementById('progress-fill').style.width = percent + '%';
                document.getElementById('status-message').textContent = message;
            }

            function showResult(success, message, stats) {
                document.getElementById('progress-container').style.display = 'none';
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').className = success ? '' : 'error';
                document.getElementById('result-title').textContent = success ? 'Processing Complete' : 'Error';
                document.getElementById('result-message').textContent = message;

                if (stats) {
                    document.getElementById('stats-grid').style.display = 'grid';
                    document.getElementById('stat-frames').textContent = stats.frames || 0;
                    document.getElementById('stat-visitors').textContent = stats.visitors || 0;
                    document.getElementById('stat-events').textContent = stats.events || 0;
                } else {
                    document.getElementById('stats-grid').style.display = 'none';
                }

                setButtonsEnabled(true);
            }

            function viewDashboard() {
                window.location.href = '/analytics-dashboard/' + venueId;
            }

            async function checkStatus() {
                if (!currentJobId) return;

                try {
                    const response = await fetch('/process/status/' + currentJobId);
                    const data = await response.json();

                    if (data.status === 'processing') {
                        const percent = data.frames_to_process > 0
                            ? Math.round((data.current_frame / data.frames_to_process) * 100)
                            : 0;
                        updateProgress(percent, data.message);
                    } else if (data.status === 'loading_model' || data.status === 'opening_video') {
                        updateProgress(5, data.message);
                    } else if (data.status === 'generating_events') {
                        updateProgress(90, data.message);
                    } else if (data.status === 'saving_events') {
                        updateProgress(95, data.message);
                    } else if (data.status === 'completed') {
                        clearInterval(statusInterval);
                        updateProgress(100, 'Complete!');
                        setTimeout(() => {
                            showResult(true, data.message, {
                                frames: data.current_frame || data.frames_to_process,
                                visitors: data.unique_visitors,
                                events: data.total_events
                            });
                        }, 500);
                    } else if (data.status === 'error') {
                        clearInterval(statusInterval);
                        showResult(false, data.message);
                    }
                } catch (e) {
                    console.error('Status check failed:', e);
                }
            }

            async function processYouTube() {
                const url = document.getElementById('youtube-url').value.trim();
                const venueData = getVenueData();
                venueId = venueData.venue_id;

                if (!url) {
                    alert('Please enter a YouTube URL');
                    return;
                }

                showProgress();
                updateProgress(0, 'Downloading video from YouTube...');

                try {
                    const response = await fetch('/process/youtube', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            url: url,
                            venue_id: venueId,
                            venue_name: venueData.venue_name,
                            latitude: venueData.latitude,
                            longitude: venueData.longitude,
                            city: venueData.city,
                            country: venueData.country,
                            venue_type: venueData.venue_type
                        })
                    });

                    const data = await response.json();

                    if (data.job_id) {
                        currentJobId = data.job_id;
                        statusInterval = setInterval(checkStatus, 1000);
                    } else {
                        showResult(false, data.detail || 'Failed to start processing');
                    }
                } catch (e) {
                    showResult(false, 'Error: ' + e.message);
                }
            }

            async function processUpload() {
                const fileInput = document.getElementById('video-file');
                const venueData = getVenueData();
                venueId = venueData.venue_id;

                if (!fileInput.files || !fileInput.files[0]) {
                    alert('Please select a video file');
                    return;
                }

                showProgress();
                updateProgress(0, 'Uploading video...');

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('venue_id', venueId);

                try {
                    const response = await fetch('/process/upload', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (data.job_id) {
                        currentJobId = data.job_id;
                        updateProgress(5, 'Processing started...');
                        statusInterval = setInterval(checkStatus, 1000);
                    } else {
                        showResult(false, data.detail || 'Failed to start processing');
                    }
                } catch (e) {
                    showResult(false, 'Error: ' + e.message);
                }
            }
        </script>
        </div>
    </body>
    </html>
    """


@app.post("/process/youtube")
async def process_youtube(data: dict, background_tasks: BackgroundTasks):
    """Process a YouTube video with optional venue location."""
    url = data.get("url")
    venue_id = data.get("venue_id", "demo_venue")

    # Extract venue location data
    venue_name = data.get("venue_name")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    city = data.get("city")
    country = data.get("country")
    venue_type = data.get("venue_type")

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    # Create or update venue with location if provided
    if latitude is not None and longitude is not None:
        h3_zone = lat_long_to_h3(latitude, longitude)

        # Check if venue exists
        existing = await database.fetch_one(
            sqlalchemy.select(venues.c.id).where(venues.c.id == venue_id)
        )

        if existing:
            # Update existing venue
            await database.execute(
                venues.update().where(venues.c.id == venue_id).values(
                    name=venue_name or existing.get("name"),
                    latitude=latitude,
                    longitude=longitude,
                    h3_zone=h3_zone,
                    city=city,
                    country=country,
                    venue_type=venue_type
                )
            )
        else:
            # Create new venue
            import secrets
            await database.execute(
                venues.insert().values(
                    id=venue_id,
                    name=venue_name or venue_id,
                    api_key=secrets.token_hex(32),
                    latitude=latitude,
                    longitude=longitude,
                    h3_zone=h3_zone,
                    city=city,
                    country=country,
                    venue_type=venue_type
                )
            )

    job_id = str(uuid.uuid4())[:8]

    processing_jobs[job_id] = {
        "status": "downloading",
        "message": "Downloading video from YouTube...",
        "venue_id": venue_id,
        "current_frame": 0,
        "total_frames": 0
    }

    def download_and_process():
        try:
            load_video_deps()

            # Download to temp file
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "video")

            processing_jobs[job_id]["message"] = "Downloading from YouTube..."

            video_path = download_youtube_video(url, output_path)

            processing_jobs[job_id]["message"] = "Download complete, starting analysis..."

            # Process the video
            process_video_file(job_id, video_path, venue_id, DATABASE_URL)

            # Cleanup
            try:
                os.remove(video_path)
                os.rmdir(temp_dir)
            except:
                pass

        except Exception as e:
            processing_jobs[job_id]["status"] = "error"
            processing_jobs[job_id]["message"] = str(e)

    thread = threading.Thread(target=download_and_process)
    thread.start()

    return {"job_id": job_id, "status": "started"}


@app.post("/process/upload")
async def process_upload(
    file: UploadFile = File(...),
    venue_id: str = Form(default="demo_venue")
):
    """Process an uploaded video file."""
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename or "video.mp4")

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    processing_jobs[job_id] = {
        "status": "starting",
        "message": "Starting video analysis...",
        "venue_id": venue_id,
        "current_frame": 0,
        "total_frames": 0
    }

    def process_and_cleanup():
        try:
            process_video_file(job_id, temp_path, venue_id, DATABASE_URL)
        finally:
            try:
                os.remove(temp_path)
                os.rmdir(temp_dir)
            except:
                pass

    thread = threading.Thread(target=process_and_cleanup)
    thread.start()

    return {"job_id": job_id, "status": "started"}


@app.get("/process/status/{job_id}")
async def get_process_status(job_id: str):
    """Get the status of a processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return processing_jobs[job_id]


# =============================================================================
# PHASE 4: MASS UPLOAD / BATCH PROCESSING
# =============================================================================

# Background queue processor state
_queue_processor_running = False
_queue_lock = threading.Lock()

async def _create_job_in_db(job_id: str, venue_id: str, video_source: str, video_name: str, priority: int = 0):
    """Create a job record in the database."""
    query = jobs.insert().values(
        id=job_id,
        venue_id=venue_id,
        status="pending",
        video_source=video_source,
        video_name=video_name,
        created_at=datetime.utcnow(),
        priority=priority
    )
    await database.execute(query)

async def _update_job_status(job_id: str, **kwargs):
    """Update job status in database."""
    query = jobs.update().where(jobs.c.id == job_id).values(**kwargs)
    await database.execute(query)

def _process_queue_sync():
    """Background worker that processes the job queue (runs in thread)."""
    global _queue_processor_running

    import asyncio

    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Create a new database connection for this thread
    import databases
    db = databases.Database(DATABASE_URL)
    loop.run_until_complete(db.connect())

    try:
        while True:
            # Get next pending job (highest priority first, then oldest)
            query = sqlalchemy.select(jobs).where(
                jobs.c.status == "pending"
            ).order_by(
                jobs.c.priority.desc(),
                jobs.c.created_at.asc()
            ).limit(1)

            job = loop.run_until_complete(db.fetch_one(query))

            if not job:
                # No jobs to process, stop the worker
                with _queue_lock:
                    _queue_processor_running = False
                break

            job_id = job["id"]
            venue_id = job["venue_id"]
            video_source = job["video_source"]

            # Mark as processing
            update_query = jobs.update().where(jobs.c.id == job_id).values(
                status="processing",
                started_at=datetime.utcnow()
            )
            loop.run_until_complete(db.execute(update_query))

            # Update in-memory tracker too
            processing_jobs[job_id] = {
                "status": "processing",
                "message": "Processing video...",
                "venue_id": venue_id,
                "current_frame": 0,
                "total_frames": 0
            }

            try:
                # Process the video
                process_video_file(job_id, video_source, venue_id, DATABASE_URL)

                # Get final stats
                final_status = processing_jobs.get(job_id, {})
                visitors = final_status.get("visitors_detected", 0)

                # Mark as completed
                update_query = jobs.update().where(jobs.c.id == job_id).values(
                    status="completed",
                    completed_at=datetime.utcnow(),
                    progress=100,
                    visitors_detected=visitors
                )
                loop.run_until_complete(db.execute(update_query))

            except Exception as e:
                # Mark as failed
                update_query = jobs.update().where(jobs.c.id == job_id).values(
                    status="failed",
                    completed_at=datetime.utcnow(),
                    error_message=str(e)
                )
                loop.run_until_complete(db.execute(update_query))

                processing_jobs[job_id] = {
                    "status": "failed",
                    "message": f"Error: {str(e)}",
                    "venue_id": venue_id
                }

            # Clean up temp file if it exists
            if os.path.exists(video_source):
                try:
                    os.remove(video_source)
                    parent = os.path.dirname(video_source)
                    if parent and os.path.isdir(parent) and not os.listdir(parent):
                        os.rmdir(parent)
                except:
                    pass

    finally:
        loop.run_until_complete(db.disconnect())
        loop.close()

def _ensure_queue_processor_running():
    """Start the queue processor if not already running."""
    global _queue_processor_running

    with _queue_lock:
        if not _queue_processor_running:
            _queue_processor_running = True
            thread = threading.Thread(target=_process_queue_sync, daemon=True)
            thread.start()

@app.post("/api/batch/upload")
async def batch_upload(
    files: List[UploadFile] = File(...),
    venue_id: str = Form(default="demo_venue"),
    priority: int = Form(default=0)
):
    """
    Upload multiple videos for batch processing.

    Returns list of job IDs for tracking progress.
    """
    created_jobs = []

    for file in files:
        job_id = str(uuid.uuid4())

        # Save uploaded file to temp location
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename or "video.mp4")

        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Create job in database
        await _create_job_in_db(
            job_id=job_id,
            venue_id=venue_id,
            video_source=temp_path,
            video_name=file.filename or "video.mp4",
            priority=priority
        )

        created_jobs.append({
            "job_id": job_id,
            "filename": file.filename,
            "status": "pending"
        })

    # Start queue processor if not running
    _ensure_queue_processor_running()

    return {
        "message": f"Queued {len(created_jobs)} videos for processing",
        "jobs": created_jobs
    }

@app.post("/api/batch/url")
async def batch_url(
    urls: List[str],
    venue_id: str = "demo_venue",
    priority: int = 0
):
    """
    Queue multiple YouTube URLs for batch processing.
    """
    load_video_deps()

    created_jobs = []

    for url in urls:
        job_id = str(uuid.uuid4())

        # Download video
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "video.mp4")

        try:
            ydl_opts = {
                'format': 'best[height<=720]',
                'outtmpl': temp_path,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_name = info.get('title', url)[:100]
        except Exception as e:
            created_jobs.append({
                "url": url,
                "status": "failed",
                "error": str(e)
            })
            continue

        # Create job in database
        await _create_job_in_db(
            job_id=job_id,
            venue_id=venue_id,
            video_source=temp_path,
            video_name=video_name,
            priority=priority
        )

        created_jobs.append({
            "job_id": job_id,
            "url": url,
            "video_name": video_name,
            "status": "pending"
        })

    # Start queue processor if not running
    _ensure_queue_processor_running()

    return {
        "message": f"Queued {len([j for j in created_jobs if j['status'] == 'pending'])} videos for processing",
        "jobs": created_jobs
    }

@app.get("/api/batch/jobs")
async def list_jobs(
    venue_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
):
    """
    List all jobs, optionally filtered by venue or status.
    """
    query = sqlalchemy.select(jobs).order_by(jobs.c.created_at.desc()).limit(limit)

    if venue_id:
        query = query.where(jobs.c.venue_id == venue_id)
    if status:
        query = query.where(jobs.c.status == status)

    rows = await database.fetch_all(query)

    return {
        "jobs": [
            {
                "id": row["id"],
                "venue_id": row["venue_id"],
                "status": row["status"],
                "video_name": row["video_name"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                "progress": row["progress"],
                "visitors_detected": row["visitors_detected"],
                "error_message": row["error_message"],
            }
            for row in rows
        ]
    }

@app.get("/api/batch/jobs/{job_id}")
async def get_job(job_id: str):
    """Get details of a specific job."""
    query = sqlalchemy.select(jobs).where(jobs.c.id == job_id)
    row = await database.fetch_one(query)

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    # Merge with in-memory status for live progress
    live_status = processing_jobs.get(job_id, {})

    return {
        "id": row["id"],
        "venue_id": row["venue_id"],
        "status": row["status"],
        "video_name": row["video_name"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
        "progress": live_status.get("progress", row["progress"]),
        "current_frame": live_status.get("current_frame", row["frames_processed"]),
        "total_frames": live_status.get("total_frames", row["total_frames"]),
        "visitors_detected": live_status.get("visitors_detected", row["visitors_detected"]),
        "error_message": row["error_message"],
        "message": live_status.get("message", "")
    }

@app.delete("/api/batch/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending job."""
    query = sqlalchemy.select(jobs).where(jobs.c.id == job_id)
    row = await database.fetch_one(query)

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    if row["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {row['status']}")

    # Delete the job
    delete_query = jobs.delete().where(jobs.c.id == job_id)
    await database.execute(delete_query)

    # Clean up file if exists
    if row["video_source"] and os.path.exists(row["video_source"]):
        try:
            os.remove(row["video_source"])
        except:
            pass

    return {"message": "Job cancelled", "job_id": job_id}

@app.get("/api/batch/stats")
async def batch_stats():
    """Get queue statistics."""
    # Count by status
    pending = await database.fetch_val(
        sqlalchemy.select(func.count()).select_from(jobs).where(jobs.c.status == "pending")
    )
    processing = await database.fetch_val(
        sqlalchemy.select(func.count()).select_from(jobs).where(jobs.c.status == "processing")
    )
    completed = await database.fetch_val(
        sqlalchemy.select(func.count()).select_from(jobs).where(jobs.c.status == "completed")
    )
    failed = await database.fetch_val(
        sqlalchemy.select(func.count()).select_from(jobs).where(jobs.c.status == "failed")
    )

    # Total visitors from completed jobs
    total_visitors = await database.fetch_val(
        sqlalchemy.select(func.sum(jobs.c.visitors_detected)).select_from(jobs).where(jobs.c.status == "completed")
    ) or 0

    return {
        "queue": {
            "pending": pending or 0,
            "processing": processing or 0,
            "completed": completed or 0,
            "failed": failed or 0,
            "total": (pending or 0) + (processing or 0) + (completed or 0) + (failed or 0)
        },
        "total_visitors_detected": total_visitors,
        "processor_running": _queue_processor_running
    }


@app.get("/dashboard/{venue_id}", response_class=HTMLResponse)
async def dashboard(venue_id: str):
    """Dashboard to view analytics for a venue."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard - {venue_id} - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }}
            /* Navigation */
            nav {{
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }}
            nav .nav-inner {{
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }}
            nav .logo {{
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }}
            nav .logo span {{ color: #3b82f6; }}
            nav .nav-links {{ display: flex; gap: 30px; }}
            nav .nav-links a {{
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }}
            nav .nav-links a:hover {{ color: #fff; }}
            nav .nav-links a.active {{ color: #3b82f6; }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
            }}
            h1 {{ color: #fff; margin-bottom: 5px; }}
            .subtitle {{ color: #888; margin-bottom: 40px; }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            .stat-card {{
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 25px;
                position: relative;
            }}
            .stat-card.primary {{
                border-color: #0066ff;
                background: linear-gradient(135deg, #1a1a2e 0%, #1a1a1a 100%);
            }}
            .stat-value {{
                font-size: 42px;
                font-weight: bold;
                color: #fff;
                margin-bottom: 5px;
            }}
            .stat-card.primary .stat-value {{
                color: #4da6ff;
            }}
            .stat-label {{ color: #888; font-size: 14px; font-weight: 500; }}
            .stat-hint {{ color: #555; font-size: 11px; margin-top: 8px; }}
            .card {{
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 20px;
            }}
            .card h3 {{ margin-top: 0; color: #fff; }}
            .bar-chart {{
                display: flex;
                align-items: flex-end;
                height: 200px;
                gap: 8px;
                padding-top: 20px;
            }}
            .bar {{
                flex: 1;
                background: linear-gradient(180deg, #0066ff, #0044aa);
                border-radius: 4px 4px 0 0;
                min-height: 4px;
                position: relative;
            }}
            .bar-label {{
                position: absolute;
                bottom: -25px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 11px;
                color: #666;
            }}
            .pie-container {{
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 30px;
            }}
            .pie-section {{
                text-align: center;
            }}
            .pie-chart {{
                width: 150px;
                height: 150px;
                border-radius: 50%;
                margin: 0 auto 15px;
            }}
            .legend {{
                display: flex;
                justify-content: center;
                gap: 20px;
                flex-wrap: wrap;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
            }}
            .legend-color {{
                width: 12px;
                height: 12px;
                border-radius: 3px;
            }}
            a {{ color: #3b82f6; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .loading {{ text-align: center; padding: 40px; color: #888; }}
            .actions {{
                display: flex;
                gap: 15px;
                margin-bottom: 30px;
            }}
            .btn {{
                background: #3b82f6;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                cursor: pointer;
                text-decoration: none;
            }}
            .btn:hover {{ background: #0052cc; text-decoration: none; }}
            .btn-secondary {{ background: #333; }}
            .btn-secondary:hover {{ background: #444; }}
            .section-title {{
                color: #666;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 15px;
            }}
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/analytics">Analytics</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container">
        <h1>📊 Venue Analytics</h1>
        <p class="subtitle">Venue: <strong>{venue_id}</strong> | Last 7 days</p>

        <div class="actions">
            <a href="/analytics-dashboard/{venue_id}" class="btn">Full Analytics Dashboard</a>
            <a href="/process" class="btn btn-secondary">Process New Video</a>
            <a href="/report/{venue_id}" class="btn btn-secondary">Print Report</a>
        </div>

        <div id="loading" class="loading">Loading analytics...</div>

        <div id="dashboard" style="display: none;">
            <div class="section-title">Key Metrics</div>
            <div class="stats-grid">
                <div class="stat-card primary">
                    <div class="stat-value" id="unique-visitors">-</div>
                    <div class="stat-label">People Detected</div>
                    <div class="stat-hint" id="visitor-range">Unique individuals tracked</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="confidence">-</div>
                    <div class="stat-label">Confidence</div>
                    <div class="stat-hint" id="data-quality">Track quality score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avg-dwell">-</div>
                    <div class="stat-label">Avg Time in View</div>
                    <div class="stat-hint">How long people stayed visible</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="peak-hour">-</div>
                    <div class="stat-label">Peak Hour</div>
                    <div class="stat-hint">Busiest time of day</div>
                </div>
            </div>

            <div class="card" id="demographics-card">
                <h3>Demographics Breakdown</h3>
                <div id="demographics-unavailable" style="text-align: center; padding: 40px; color: #666;">
                    No faces detected in video.<br>
                    Demographics require visible faces (front-facing).
                </div>
                <div class="pie-container" id="demographics-charts" style="display: none;">
                    <div class="pie-section">
                        <h4>Gender</h4>
                        <div id="gender-chart" class="pie-chart"></div>
                        <div id="gender-legend" class="legend"></div>
                    </div>
                    <div class="pie-section">
                        <h4>Age Groups</h4>
                        <div id="age-chart" class="pie-chart"></div>
                        <div id="age-legend" class="legend"></div>
                    </div>
                </div>
                </div>
            </div>

            <div class="card">
                <h3>Traffic by Hour</h3>
                <div id="hourly-chart" class="bar-chart"></div>
            </div>
        </div>

        <script>
            const venueId = '{venue_id}';

            function createPieChart(elementId, data, colors) {{
                const container = document.getElementById(elementId);
                const total = Object.values(data).reduce((a, b) => a + b, 0);

                if (total === 0) {{
                    container.style.background = '#333';
                    return;
                }}

                let gradient = 'conic-gradient(';
                let currentAngle = 0;
                const entries = Object.entries(data);

                entries.forEach(([key, value], i) => {{
                    const angle = (value / total) * 360;
                    const color = colors[i % colors.length];
                    gradient += `${{color}} ${{currentAngle}}deg ${{currentAngle + angle}}deg`;
                    if (i < entries.length - 1) gradient += ', ';
                    currentAngle += angle;
                }});

                gradient += ')';
                container.style.background = gradient;
            }}

            function createLegend(elementId, data, colors) {{
                const container = document.getElementById(elementId);
                const total = Object.values(data).reduce((a, b) => a + b, 0);

                container.innerHTML = Object.entries(data).map(([key, value], i) => {{
                    const percent = total > 0 ? Math.round((value / total) * 100) : 0;
                    return `
                        <div class="legend-item">
                            <div class="legend-color" style="background: ${{colors[i % colors.length]}}"></div>
                            <span>${{key}}: ${{percent}}%</span>
                        </div>
                    `;
                }}).join('');
            }}

            function createBarChart(elementId, data) {{
                const container = document.getElementById(elementId);
                const maxValue = Math.max(...data.map(d => d.visitors), 1);

                container.innerHTML = data.map(d => {{
                    const height = (d.visitors / maxValue) * 100;
                    return `
                        <div class="bar" style="height: ${{Math.max(height, 2)}}%">
                            <span class="bar-label">${{d.hour}}:00</span>
                        </div>
                    `;
                }}).join('');
            }}

            async function loadData() {{
                try {{
                    const [statsRes, hourlyRes] = await Promise.all([
                        fetch(`/analytics/${{venueId}}?days=7`),
                        fetch(`/analytics/${{venueId}}/hourly`)
                    ]);

                    const stats = await statsRes.json();
                    const hourly = await hourlyRes.json();

                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('dashboard').style.display = 'block';

                    // Update stats with clear labels
                    document.getElementById('unique-visitors').textContent = stats.unique_visitors.toLocaleString();
                    document.getElementById('avg-dwell').textContent = stats.avg_dwell_minutes + ' min';
                    document.getElementById('peak-hour').textContent = stats.peak_hour !== null ? stats.peak_hour + ':00' : '-';

                    // Confidence metrics
                    if (stats.confidence_level !== null) {{
                        document.getElementById('confidence').textContent = Math.round(stats.confidence_level * 100) + '%';
                        document.getElementById('data-quality').textContent = 'Data quality: ' + (stats.data_quality || 'unknown');
                        if (stats.visitor_range) {{
                            document.getElementById('visitor-range').textContent =
                                `Range: ${{stats.visitor_range.low}} - ${{stats.visitor_range.high}} (95% CI)`;
                        }}
                    }} else {{
                        document.getElementById('confidence').textContent = '-';
                        document.getElementById('data-quality').textContent = 'No track data';
                    }}

                    // Demographics - only show if we have real data
                    const hasGenderData = Object.keys(stats.gender_split).some(k => k !== null && k !== 'null');
                    const hasAgeData = Object.keys(stats.age_distribution).some(k => k !== null && k !== 'null');

                    if (hasGenderData || hasAgeData) {{
                        document.getElementById('demographics-unavailable').style.display = 'none';
                        document.getElementById('demographics-charts').style.display = 'flex';

                        // Gender chart
                        const genderColors = ['#0066ff', '#ff6b9d'];
                        createPieChart('gender-chart', stats.gender_split, genderColors);
                        createLegend('gender-legend', stats.gender_split, genderColors);

                        // Age chart
                        const ageColors = ['#00cc88', '#0066ff', '#ff9500', '#ff6b6b'];
                        createPieChart('age-chart', stats.age_distribution, ageColors);
                        createLegend('age-legend', stats.age_distribution, ageColors);
                    }} else {{
                        document.getElementById('demographics-unavailable').style.display = 'block';
                        document.getElementById('demographics-charts').style.display = 'none';
                    }}

                    // Hourly chart
                    createBarChart('hourly-chart', hourly.hourly);

                }} catch (e) {{
                    document.getElementById('loading').textContent = 'Error loading data: ' + e.message;
                }}
            }}

            loadData();
        </script>
        </div>
    </body>
    </html>
    """


# =============================================================================
# COMPREHENSIVE ANALYTICS DASHBOARD
# =============================================================================

@app.get("/analytics-dashboard/{venue_id}", response_class=HTMLResponse)
async def analytics_dashboard(venue_id: str):
    """
    Comprehensive analytics dashboard with filters, charts, and data tables.
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics Dashboard - {venue_id} - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }}
            /* Navigation */
            nav {{
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }}
            nav .nav-inner {{
                max-width: 1600px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }}
            nav .logo {{
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }}
            nav .logo span {{ color: #3b82f6; }}
            nav .nav-links {{ display: flex; gap: 30px; }}
            nav .nav-links a {{
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }}
            nav .nav-links a:hover {{ color: #fff; }}
            nav .nav-links a.active {{ color: #3b82f6; }}

            .container {{
                max-width: 1600px;
                margin: 0 auto;
                padding: 30px 20px;
            }}

            /* Header */
            .page-header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 30px;
                flex-wrap: wrap;
                gap: 20px;
            }}
            .page-header h1 {{ color: #fff; font-size: 28px; }}
            .page-header .subtitle {{ color: #888; margin-top: 5px; }}

            /* Filters */
            .filters {{
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 25px;
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                align-items: flex-end;
            }}
            .filter-group {{
                display: flex;
                flex-direction: column;
                gap: 6px;
            }}
            .filter-group label {{
                font-size: 12px;
                color: #888;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .filter-group select, .filter-group input {{
                background: #0a0a0a;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 10px 14px;
                color: #fff;
                font-size: 14px;
                min-width: 150px;
            }}
            .filter-group select:focus, .filter-group input:focus {{
                border-color: #3b82f6;
                outline: none;
            }}
            .btn {{
                background: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
                cursor: pointer;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }}
            .btn:hover {{ background: #2563eb; }}
            .btn-secondary {{ background: #333; }}
            .btn-secondary:hover {{ background: #444; }}
            .btn-sm {{ padding: 6px 12px; font-size: 12px; }}

            /* KPI Cards */
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 25px;
            }}
            .kpi-card {{
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 20px;
            }}
            .kpi-card.highlight {{
                border-color: #3b82f6;
                background: linear-gradient(135deg, #1a1a2e 0%, #1a1a1a 100%);
            }}
            .kpi-value {{
                font-size: 36px;
                font-weight: bold;
                color: #fff;
            }}
            .kpi-card.highlight .kpi-value {{ color: #60a5fa; }}
            .kpi-label {{
                color: #888;
                font-size: 13px;
                margin-top: 5px;
            }}
            .kpi-change {{
                font-size: 12px;
                margin-top: 8px;
                display: flex;
                align-items: center;
                gap: 4px;
            }}
            .kpi-change.positive {{ color: #22c55e; }}
            .kpi-change.negative {{ color: #ef4444; }}
            .kpi-change.neutral {{ color: #888; }}

            /* Charts Grid */
            .charts-grid {{
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
                margin-bottom: 25px;
            }}
            @media (max-width: 1200px) {{
                .charts-grid {{ grid-template-columns: 1fr; }}
            }}

            .card {{
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 20px;
            }}
            .card-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }}
            .card-header h3 {{
                color: #fff;
                font-size: 16px;
                font-weight: 600;
            }}
            .card-header .tabs {{
                display: flex;
                gap: 5px;
            }}
            .card-header .tab {{
                padding: 6px 12px;
                background: transparent;
                border: 1px solid #333;
                border-radius: 6px;
                color: #888;
                font-size: 12px;
                cursor: pointer;
            }}
            .card-header .tab.active {{
                background: #3b82f6;
                border-color: #3b82f6;
                color: #fff;
            }}

            .chart-container {{
                position: relative;
                height: 300px;
            }}

            /* Side Charts */
            .side-charts {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .mini-chart {{
                height: 180px;
            }}

            /* Heatmap */
            .heatmap-container {{
                margin-bottom: 25px;
            }}
            .heatmap {{
                display: grid;
                grid-template-columns: 60px repeat(24, 1fr);
                gap: 2px;
                font-size: 11px;
            }}
            .heatmap-header {{
                color: #666;
                text-align: center;
                padding: 5px 0;
            }}
            .heatmap-row-label {{
                color: #888;
                display: flex;
                align-items: center;
                padding-right: 10px;
            }}
            .heatmap-cell {{
                aspect-ratio: 1;
                border-radius: 3px;
                min-height: 20px;
                cursor: pointer;
                transition: transform 0.1s;
            }}
            .heatmap-cell:hover {{
                transform: scale(1.2);
                z-index: 10;
            }}
            .heatmap-legend {{
                display: flex;
                justify-content: flex-end;
                align-items: center;
                gap: 10px;
                margin-top: 15px;
                font-size: 12px;
                color: #888;
            }}
            .heatmap-legend-gradient {{
                width: 150px;
                height: 12px;
                border-radius: 6px;
                background: linear-gradient(90deg, #1a1a2e, #1e3a5f, #2563eb, #3b82f6, #60a5fa);
            }}

            /* Data Table */
            .table-container {{
                overflow-x: auto;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #333;
            }}
            th {{
                color: #888;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 600;
                cursor: pointer;
                user-select: none;
            }}
            th:hover {{ color: #fff; }}
            th.sorted {{ color: #3b82f6; }}
            td {{ color: #e0e0e0; }}
            tr:hover td {{ background: #222; }}

            .progress-bar {{
                width: 100%;
                height: 6px;
                background: #333;
                border-radius: 3px;
                overflow: hidden;
            }}
            .progress-bar-fill {{
                height: 100%;
                border-radius: 3px;
                transition: width 0.3s;
            }}

            .badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 500;
            }}
            .badge-engaged {{ background: #22c55e33; color: #22c55e; }}
            .badge-browsing {{ background: #3b82f633; color: #3b82f6; }}
            .badge-waiting {{ background: #f59e0b33; color: #f59e0b; }}
            .badge-passing {{ background: #6b728033; color: #9ca3af; }}

            /* Loading */
            .loading {{
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 60px;
                color: #888;
            }}
            .spinner {{
                width: 40px;
                height: 40px;
                border: 3px solid #333;
                border-top-color: #3b82f6;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 15px;
            }}
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

            /* Tooltip */
            .tooltip {{
                position: absolute;
                background: #222;
                border: 1px solid #444;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
                white-space: nowrap;
            }}

            /* Comparison Mode */
            .comparison-toggle {{
                display: flex;
                align-items: center;
                gap: 8px;
                color: #888;
                font-size: 13px;
            }}
            .comparison-toggle input {{
                width: 18px;
                height: 18px;
            }}

            /* Export dropdown */
            .export-dropdown {{
                position: relative;
            }}
            .export-menu {{
                position: absolute;
                top: 100%;
                right: 0;
                background: #222;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 8px 0;
                margin-top: 5px;
                display: none;
                min-width: 150px;
                z-index: 100;
            }}
            .export-menu.show {{ display: block; }}
            .export-menu a {{
                display: block;
                padding: 10px 15px;
                color: #e0e0e0;
                text-decoration: none;
                font-size: 13px;
            }}
            .export-menu a:hover {{ background: #333; }}
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/analytics">Analytics</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container">
            <!-- Header -->
            <div class="page-header">
                <div>
                    <h1>Analytics Dashboard</h1>
                    <p class="subtitle">Venue: <strong>{venue_id}</strong></p>
                </div>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <label class="comparison-toggle">
                        <input type="checkbox" id="compare-toggle">
                        Compare to previous period
                    </label>
                    <div class="export-dropdown">
                        <button class="btn btn-secondary" onclick="toggleExportMenu()">
                            Export
                            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                                <path d="M2 4l4 4 4-4"/>
                            </svg>
                        </button>
                        <div class="export-menu" id="export-menu">
                            <a href="/analytics/{venue_id}/export?format=json" target="_blank">Export JSON</a>
                            <a href="/analytics/{venue_id}/export?format=csv">Download CSV</a>
                            <a href="/report/{venue_id}" target="_blank">Print Report</a>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Filters -->
            <div class="filters">
                <div class="filter-group">
                    <label>Date Range</label>
                    <select id="filter-days">
                        <option value="1">Today</option>
                        <option value="7" selected>Last 7 days</option>
                        <option value="14">Last 14 days</option>
                        <option value="30">Last 30 days</option>
                        <option value="90">Last 90 days</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Zone</label>
                    <select id="filter-zone">
                        <option value="">All Zones</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Gender</label>
                    <select id="filter-gender">
                        <option value="">All</option>
                        <option value="M">Male</option>
                        <option value="F">Female</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Age Group</label>
                    <select id="filter-age">
                        <option value="">All Ages</option>
                        <option value="18-24">18-24</option>
                        <option value="25-34">25-34</option>
                        <option value="35-44">35-44</option>
                        <option value="45-54">45-54</option>
                        <option value="55+">55+</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Behavior</label>
                    <select id="filter-behavior">
                        <option value="">All Behaviors</option>
                        <option value="engaged">Engaged</option>
                        <option value="browsing">Browsing</option>
                        <option value="waiting">Waiting</option>
                        <option value="passing">Passing</option>
                    </select>
                </div>
                <button class="btn" onclick="applyFilters()">Apply Filters</button>
                <button class="btn btn-secondary" onclick="resetFilters()">Reset</button>
            </div>

            <!-- KPI Cards -->
            <div class="kpi-grid" id="kpi-grid">
                <div class="loading"><div class="spinner"></div> Loading...</div>
            </div>

            <!-- Main Charts -->
            <div class="charts-grid">
                <div class="card">
                    <div class="card-header">
                        <h3>Traffic Trend</h3>
                        <div class="tabs">
                            <button class="tab active" data-view="daily">Daily</button>
                            <button class="tab" data-view="hourly">Hourly</button>
                        </div>
                    </div>
                    <div class="chart-container">
                        <canvas id="traffic-chart"></canvas>
                    </div>
                </div>
                <div class="side-charts">
                    <div class="card">
                        <div class="card-header">
                            <h3>Demographics</h3>
                        </div>
                        <div class="mini-chart">
                            <canvas id="demographics-chart"></canvas>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header">
                            <h3>Behavior Mix</h3>
                        </div>
                        <div class="mini-chart">
                            <canvas id="behavior-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Heatmap -->
            <div class="card heatmap-container">
                <div class="card-header">
                    <h3>Weekly Traffic Heatmap</h3>
                    <span style="color: #888; font-size: 13px;">Visitors by day and hour</span>
                </div>
                <div class="heatmap" id="heatmap"></div>
                <div class="heatmap-legend">
                    <span>Low</span>
                    <div class="heatmap-legend-gradient"></div>
                    <span>High</span>
                </div>
            </div>

            <!-- Zone Performance Table -->
            <div class="card">
                <div class="card-header">
                    <h3>Zone Performance</h3>
                    <button class="btn btn-sm btn-secondary" onclick="exportTable()">Export CSV</button>
                </div>
                <div class="table-container">
                    <table id="zone-table">
                        <thead>
                            <tr>
                                <th onclick="sortTable(0)">Zone</th>
                                <th onclick="sortTable(1)">Visitors</th>
                                <th onclick="sortTable(2)">Avg Dwell</th>
                                <th onclick="sortTable(3)">Engagement</th>
                                <th>Traffic Share</th>
                                <th onclick="sortTable(5)">Top Behavior</th>
                            </tr>
                        </thead>
                        <tbody id="zone-tbody">
                            <tr><td colspan="6" class="loading"><div class="spinner"></div> Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Engagement by Hour -->
            <div class="card" style="margin-top: 20px;">
                <div class="card-header">
                    <h3>Engagement by Hour</h3>
                </div>
                <div class="chart-container">
                    <canvas id="engagement-chart"></canvas>
                </div>
            </div>
        </div>

        <script>
            const venueId = '{venue_id}';
            let currentDays = 7;
            let allData = {{}};
            let trafficChart, demographicsChart, behaviorChart, engagementChart;

            // Chart.js defaults
            Chart.defaults.color = '#888';
            Chart.defaults.borderColor = '#333';

            // Initialize
            document.addEventListener('DOMContentLoaded', () => {{
                loadAllData();
                setupEventListeners();
            }});

            function setupEventListeners() {{
                // Tab switching
                document.querySelectorAll('.tab').forEach(tab => {{
                    tab.addEventListener('click', (e) => {{
                        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                        e.target.classList.add('active');
                        updateTrafficChart(e.target.dataset.view);
                    }});
                }});

                // Comparison toggle
                document.getElementById('compare-toggle').addEventListener('change', (e) => {{
                    loadAllData();
                }});
            }}

            async function loadAllData() {{
                const days = document.getElementById('filter-days').value;
                currentDays = parseInt(days);

                try {{
                    const [summary, hourly, demographics, zones, behavior, behaviorHourly, heatmap] = await Promise.all([
                        fetch(`/analytics/${{venueId}}/summary?days=${{days}}`).then(r => r.json()),
                        fetch(`/analytics/${{venueId}}/hourly`).then(r => r.json()),
                        fetch(`/analytics/${{venueId}}/demographics?days=${{days}}`).then(r => r.json()),
                        fetch(`/analytics/${{venueId}}/zones?days=${{days}}`).then(r => r.json()),
                        fetch(`/analytics/${{venueId}}/behavior?days=${{days}}`).then(r => r.json()),
                        fetch(`/analytics/${{venueId}}/behavior/hourly?days=${{days}}`).then(r => r.json()),
                        fetch(`/analytics/${{venueId}}/heatmap?weeks=${{Math.ceil(days/7)}}`).then(r => r.json())
                    ]);

                    allData = {{ summary, hourly, demographics, zones, behavior, behaviorHourly, heatmap }};

                    renderKPIs(summary);
                    renderTrafficChart(hourly);
                    renderDemographicsChart(demographics);
                    renderBehaviorChart(behavior);
                    renderHeatmap(heatmap);
                    renderZoneTable(zones, behavior);
                    renderEngagementChart(behaviorHourly);
                    populateZoneFilter(zones);
                }} catch (error) {{
                    console.error('Error loading data:', error);
                }}
            }}

            function renderKPIs(summary) {{
                const current = summary.current || {{}};
                const change = summary.change || {{}};
                const showComparison = document.getElementById('compare-toggle').checked;

                const kpis = [
                    {{
                        value: current.unique_visitors || 0,
                        label: 'Unique Visitors',
                        change: change.visitors_percent,
                        highlight: true
                    }},
                    {{
                        value: (current.return_rate_percent || 0) + '%',
                        label: 'Return Rate',
                        change: null
                    }},
                    {{
                        value: (current.avg_dwell_minutes || 0) + ' min',
                        label: 'Avg Dwell Time',
                        change: change.dwell_minutes ? (change.dwell_minutes > 0 ? '+' : '') + change.dwell_minutes + ' min' : null
                    }},
                    {{
                        value: current.peak_hour !== null ? current.peak_hour + ':00' : '-',
                        label: 'Peak Hour',
                        change: null
                    }},
                    {{
                        value: current.avg_engagement !== null ? current.avg_engagement : '-',
                        label: 'Avg Engagement',
                        change: null
                    }},
                    {{
                        value: (current.engaged_percent || 0) + '%',
                        label: 'Highly Engaged',
                        change: null
                    }}
                ];

                document.getElementById('kpi-grid').innerHTML = kpis.map(kpi => `
                    <div class="kpi-card ${{kpi.highlight ? 'highlight' : ''}}">
                        <div class="kpi-value">${{kpi.value}}</div>
                        <div class="kpi-label">${{kpi.label}}</div>
                        ${{showComparison && kpi.change !== null ? `
                            <div class="kpi-change ${{kpi.change > 0 ? 'positive' : kpi.change < 0 ? 'negative' : 'neutral'}}">
                                ${{kpi.change > 0 ? '↑' : kpi.change < 0 ? '↓' : '→'}} ${{Math.abs(kpi.change)}}${{typeof kpi.change === 'number' ? '%' : ''}} vs previous
                            </div>
                        ` : ''}}
                    </div>
                `).join('');
            }}

            function renderTrafficChart(hourly) {{
                const ctx = document.getElementById('traffic-chart').getContext('2d');

                if (trafficChart) trafficChart.destroy();

                const data = hourly.hourly || hourly.hourly_breakdown || [];

                trafficChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: data.map(d => d.hour + ':00'),
                        datasets: [{{
                            label: 'Visitors',
                            data: data.map(d => d.visitors),
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 4,
                            pointHoverRadius: 6
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                grid: {{ color: '#222' }}
                            }},
                            x: {{
                                grid: {{ display: false }}
                            }}
                        }}
                    }}
                }});
            }}

            function renderDemographicsChart(demographics) {{
                const ctx = document.getElementById('demographics-chart').getContext('2d');

                if (demographicsChart) demographicsChart.destroy();

                const genderData = demographics.gender_split || (demographics.current && demographics.current.gender) || {{}};
                const labels = Object.keys(genderData).filter(k => k && k !== 'null' && k !== 'undefined');
                const values = labels.map(k => genderData[k]);

                demographicsChart = new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: labels.map(l => l === 'M' ? 'Male' : l === 'F' ? 'Female' : l),
                        datasets: [{{
                            data: values,
                            backgroundColor: ['#3b82f6', '#ec4899', '#8b5cf6'],
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom',
                                labels: {{ padding: 15, usePointStyle: true }}
                            }}
                        }}
                    }}
                }});
            }}

            function renderBehaviorChart(behavior) {{
                const ctx = document.getElementById('behavior-chart').getContext('2d');

                if (behaviorChart) behaviorChart.destroy();

                const types = behavior.behavior_types || {{}};
                const labels = Object.keys(types);
                const values = Object.values(types);

                const colors = {{
                    'engaged': '#22c55e',
                    'browsing': '#3b82f6',
                    'waiting': '#f59e0b',
                    'passing': '#6b7280'
                }};

                behaviorChart = new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
                        datasets: [{{
                            data: values,
                            backgroundColor: labels.map(l => colors[l] || '#888'),
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom',
                                labels: {{ padding: 15, usePointStyle: true }}
                            }}
                        }}
                    }}
                }});
            }}

            function renderHeatmap(heatmap) {{
                const container = document.getElementById('heatmap');
                const data = heatmap.heatmap || {{}};
                const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
                const maxCount = heatmap.max_count || 1;

                let html = '<div class="heatmap-header"></div>';
                for (let h = 0; h < 24; h++) {{
                    html += `<div class="heatmap-header">${{h}}</div>`;
                }}

                days.forEach(day => {{
                    html += `<div class="heatmap-row-label">${{day}}</div>`;
                    for (let h = 0; h < 24; h++) {{
                        const count = (data[day] && data[day][h]) || 0;
                        const intensity = count / maxCount;
                        const color = getHeatmapColor(intensity);
                        html += `<div class="heatmap-cell" style="background: ${{color}}" title="${{day}} ${{h}}:00 - ${{count}} visitors"></div>`;
                    }}
                }});

                container.innerHTML = html;
            }}

            function getHeatmapColor(intensity) {{
                if (intensity === 0) return '#1a1a2e';
                if (intensity < 0.25) return '#1e3a5f';
                if (intensity < 0.5) return '#2563eb';
                if (intensity < 0.75) return '#3b82f6';
                return '#60a5fa';
            }}

            function renderZoneTable(zones, behavior) {{
                const tbody = document.getElementById('zone-tbody');
                const zoneData = zones.zones || [];
                const behaviorZones = allData.behaviorHourly || {{}};

                if (zoneData.length === 0) {{
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666; padding: 40px;">No zone data available</td></tr>';
                    return;
                }}

                const totalVisitors = zoneData.reduce((sum, z) => sum + z.visitors, 0);

                tbody.innerHTML = zoneData.map(zone => {{
                    const share = totalVisitors > 0 ? (zone.visitors / totalVisitors * 100).toFixed(1) : 0;
                    const engagement = zone.engagement || zone.avg_engagement || '-';
                    const topBehavior = zone.top_behavior || 'browsing';

                    return `
                        <tr>
                            <td><strong>${{zone.zone}}</strong></td>
                            <td>${{zone.visitors.toLocaleString()}}</td>
                            <td>${{zone.avg_dwell_minutes || (zone.avg_dwell_seconds / 60).toFixed(1)}} min</td>
                            <td>${{typeof engagement === 'number' ? engagement.toFixed(1) : engagement}}</td>
                            <td>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <div class="progress-bar" style="width: 100px;">
                                        <div class="progress-bar-fill" style="width: ${{share}}%; background: #3b82f6;"></div>
                                    </div>
                                    <span>${{share}}%</span>
                                </div>
                            </td>
                            <td><span class="badge badge-${{topBehavior}}">${{topBehavior}}</span></td>
                        </tr>
                    `;
                }}).join('');
            }}

            function renderEngagementChart(behaviorHourly) {{
                const ctx = document.getElementById('engagement-chart').getContext('2d');

                if (engagementChart) engagementChart.destroy();

                const data = behaviorHourly.hourly_engagement || [];

                engagementChart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: data.map(d => d.hour + ':00'),
                        datasets: [{{
                            label: 'Avg Engagement',
                            data: data.map(d => d.avg_engagement),
                            backgroundColor: data.map(d => {{
                                if (d.avg_engagement >= 70) return '#22c55e';
                                if (d.avg_engagement >= 50) return '#3b82f6';
                                return '#6b7280';
                            }}),
                            borderRadius: 4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 100,
                                grid: {{ color: '#222' }},
                                title: {{ display: true, text: 'Engagement Score' }}
                            }},
                            x: {{
                                grid: {{ display: false }}
                            }}
                        }}
                    }}
                }});
            }}

            function populateZoneFilter(zones) {{
                const select = document.getElementById('filter-zone');
                const zoneData = zones.zones || [];

                select.innerHTML = '<option value="">All Zones</option>' +
                    zoneData.map(z => `<option value="${{z.zone}}">${{z.zone}}</option>`).join('');
            }}

            function applyFilters() {{
                loadAllData();
            }}

            function resetFilters() {{
                document.getElementById('filter-days').value = '7';
                document.getElementById('filter-zone').value = '';
                document.getElementById('filter-gender').value = '';
                document.getElementById('filter-age').value = '';
                document.getElementById('filter-behavior').value = '';
                loadAllData();
            }}

            function toggleExportMenu() {{
                document.getElementById('export-menu').classList.toggle('show');
            }}

            // Close export menu when clicking outside
            document.addEventListener('click', (e) => {{
                if (!e.target.closest('.export-dropdown')) {{
                    document.getElementById('export-menu').classList.remove('show');
                }}
            }});

            function updateTrafficChart(view) {{
                // For now, just reload - could implement daily aggregation
                renderTrafficChart(allData.hourly);
            }}

            let sortDirection = 1;
            let sortColumn = -1;

            function sortTable(column) {{
                const tbody = document.getElementById('zone-tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));

                if (sortColumn === column) {{
                    sortDirection *= -1;
                }} else {{
                    sortDirection = 1;
                    sortColumn = column;
                }}

                rows.sort((a, b) => {{
                    let aVal = a.cells[column].textContent.trim();
                    let bVal = b.cells[column].textContent.trim();

                    // Try numeric comparison
                    const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                    const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return (aNum - bNum) * sortDirection;
                    }}

                    return aVal.localeCompare(bVal) * sortDirection;
                }});

                rows.forEach(row => tbody.appendChild(row));

                // Update header styling
                document.querySelectorAll('th').forEach((th, i) => {{
                    th.classList.toggle('sorted', i === column);
                }});
            }}

            function exportTable() {{
                const table = document.getElementById('zone-table');
                const rows = table.querySelectorAll('tr');
                let csv = [];

                rows.forEach(row => {{
                    const cols = row.querySelectorAll('th, td');
                    const rowData = Array.from(cols).map(col => {{
                        let text = col.textContent.trim().replace(/"/g, '""');
                        return `"${{text}}"`;
                    }});
                    csv.push(rowData.join(','));
                }});

                const blob = new Blob([csv.join('\\n')], {{ type: 'text/csv' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${{venueId}}_zone_performance.csv`;
                a.click();
            }}
        </script>
    </body>
    </html>
    """


# =============================================================================
# UPLOADS DASHBOARD (Phase 4)
# =============================================================================

@app.get("/uploads", response_class=HTMLResponse)
async def uploads_dashboard():
    """Dashboard to manage batch video uploads and queue."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Manager - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }
            /* Navigation */
            nav {
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            nav .nav-inner {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }
            nav .logo {
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }
            nav .logo span { color: #3b82f6; }
            nav .nav-links { display: flex; gap: 30px; }
            nav .nav-links a {
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }
            nav .nav-links a:hover { color: #fff; }
            nav .nav-links a.active { color: #3b82f6; }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            h1 { color: #fff; margin-bottom: 10px; }
            .subtitle { color: #888; margin-bottom: 30px; }

            /* Stats bar */
            .stats-bar {
                display: flex;
                gap: 20px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            .stat-card {
                background: #1a1a1a;
                border-radius: 12px;
                padding: 20px 30px;
                border: 1px solid #333;
                min-width: 140px;
            }
            .stat-value {
                font-size: 32px;
                font-weight: bold;
                color: #fff;
            }
            .stat-label { color: #888; font-size: 14px; margin-top: 5px; }
            .stat-card.pending .stat-value { color: #f59e0b; }
            .stat-card.processing .stat-value { color: #3b82f6; }
            .stat-card.completed .stat-value { color: #22c55e; }
            .stat-card.failed .stat-value { color: #ef4444; }

            /* Upload section */
            .upload-section {
                background: #1a1a1a;
                border-radius: 12px;
                padding: 30px;
                border: 1px solid #333;
                margin-bottom: 30px;
            }
            .upload-section h2 { margin-bottom: 20px; color: #fff; }

            .upload-area {
                border: 2px dashed #444;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.2s;
                margin-bottom: 20px;
            }
            .upload-area:hover, .upload-area.dragover {
                border-color: #3b82f6;
                background: rgba(59, 130, 246, 0.1);
            }
            .upload-area input { display: none; }
            .upload-icon { font-size: 48px; margin-bottom: 10px; }
            .upload-text { color: #888; }
            .upload-text strong { color: #3b82f6; }

            .upload-options {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                align-items: end;
            }
            .form-group { flex: 1; min-width: 200px; }
            .form-group label { display: block; margin-bottom: 8px; color: #888; font-size: 14px; }
            .form-group input, .form-group select {
                width: 100%;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid #333;
                background: #0a0a0a;
                color: #fff;
                font-size: 14px;
            }
            .btn {
                padding: 12px 24px;
                border-radius: 8px;
                border: none;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }
            .btn-primary { background: #3b82f6; color: white; }
            .btn-primary:hover { background: #2563eb; }
            .btn-primary:disabled { background: #555; cursor: not-allowed; }

            /* Jobs table */
            .jobs-section {
                background: #1a1a1a;
                border-radius: 12px;
                padding: 30px;
                border: 1px solid #333;
            }
            .jobs-section h2 { margin-bottom: 20px; color: #fff; }

            .jobs-table {
                width: 100%;
                border-collapse: collapse;
            }
            .jobs-table th, .jobs-table td {
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid #333;
            }
            .jobs-table th { color: #888; font-weight: 500; }
            .jobs-table tr:hover { background: rgba(255,255,255,0.02); }

            .status-badge {
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
            }
            .status-pending { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
            .status-processing { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
            .status-completed { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
            .status-failed { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

            .progress-bar {
                width: 100px;
                height: 6px;
                background: #333;
                border-radius: 3px;
                overflow: hidden;
            }
            .progress-fill {
                height: 100%;
                background: #3b82f6;
                transition: width 0.3s;
            }

            .btn-small {
                padding: 6px 12px;
                font-size: 12px;
            }
            .btn-danger { background: #ef4444; color: white; }
            .btn-danger:hover { background: #dc2626; }

            .empty-state {
                text-align: center;
                padding: 60px 20px;
                color: #666;
            }
            .empty-state .icon { font-size: 48px; margin-bottom: 15px; }

            /* Upload progress list */
            .upload-progress-list {
                margin-top: 20px;
            }
            .upload-item {
                display: flex;
                align-items: center;
                gap: 15px;
                padding: 10px;
                background: #0a0a0a;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .upload-item .filename { flex: 1; }
            .upload-item .size { color: #888; font-size: 12px; }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads" class="active">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container">
            <h1>📤 Upload Manager</h1>
            <p class="subtitle">Batch upload and process videos</p>

            <!-- Stats Bar -->
            <div class="stats-bar">
                <div class="stat-card pending">
                    <div class="stat-value" id="stat-pending">-</div>
                    <div class="stat-label">Pending</div>
                </div>
                <div class="stat-card processing">
                    <div class="stat-value" id="stat-processing">-</div>
                    <div class="stat-label">Processing</div>
                </div>
                <div class="stat-card completed">
                    <div class="stat-value" id="stat-completed">-</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-card failed">
                    <div class="stat-value" id="stat-failed">-</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="stat-visitors">-</div>
                    <div class="stat-label">Total Visitors Detected</div>
                </div>
            </div>

            <!-- Upload Section -->
            <div class="upload-section">
                <h2>Upload Videos</h2>
                <div class="upload-area" id="upload-area">
                    <input type="file" id="file-input" multiple accept="video/*">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">
                        <strong>Click to upload</strong> or drag and drop<br>
                        Multiple video files supported (MP4, MOV, AVI)
                    </div>
                </div>

                <div class="upload-options">
                    <div class="form-group">
                        <label>Venue ID</label>
                        <input type="text" id="venue-id" value="demo_venue" placeholder="Enter venue ID">
                    </div>
                    <div class="form-group">
                        <label>Priority</label>
                        <select id="priority">
                            <option value="0">Normal</option>
                            <option value="5">High</option>
                            <option value="10">Urgent</option>
                        </select>
                    </div>
                    <button class="btn btn-primary" id="upload-btn" disabled>Upload Selected Files</button>
                </div>

                <div class="upload-progress-list" id="upload-progress"></div>
            </div>

            <!-- Jobs Table -->
            <div class="jobs-section">
                <h2>Processing Queue</h2>
                <table class="jobs-table">
                    <thead>
                        <tr>
                            <th>Video</th>
                            <th>Venue</th>
                            <th>Status</th>
                            <th>Progress</th>
                            <th>Visitors</th>
                            <th>Created</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="jobs-tbody">
                        <tr>
                            <td colspan="7">
                                <div class="empty-state">
                                    <div class="icon">📭</div>
                                    <div>No jobs yet. Upload some videos to get started!</div>
                                </div>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            let selectedFiles = [];

            // File input handling
            const uploadArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-input');
            const uploadBtn = document.getElementById('upload-btn');
            const uploadProgress = document.getElementById('upload-progress');

            uploadArea.addEventListener('click', () => fileInput.click());

            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                handleFiles(e.dataTransfer.files);
            });

            fileInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });

            function handleFiles(files) {
                selectedFiles = Array.from(files).filter(f => f.type.startsWith('video/'));
                updateFileList();
            }

            function updateFileList() {
                if (selectedFiles.length === 0) {
                    uploadProgress.innerHTML = '';
                    uploadBtn.disabled = true;
                    return;
                }

                uploadBtn.disabled = false;
                uploadProgress.innerHTML = selectedFiles.map((f, i) => `
                    <div class="upload-item">
                        <span>📹</span>
                        <span class="filename">${f.name}</span>
                        <span class="size">${(f.size / 1024 / 1024).toFixed(1)} MB</span>
                    </div>
                `).join('');
            }

            uploadBtn.addEventListener('click', async () => {
                if (selectedFiles.length === 0) return;

                uploadBtn.disabled = true;
                uploadBtn.textContent = 'Uploading...';

                const formData = new FormData();
                selectedFiles.forEach(f => formData.append('files', f));
                formData.append('venue_id', document.getElementById('venue-id').value);
                formData.append('priority', document.getElementById('priority').value);

                try {
                    const resp = await fetch('/api/batch/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await resp.json();

                    if (resp.ok) {
                        alert(`Queued ${data.jobs.length} videos for processing!`);
                        selectedFiles = [];
                        updateFileList();
                        fileInput.value = '';
                        loadJobs();
                        loadStats();
                    } else {
                        alert('Upload failed: ' + (data.detail || 'Unknown error'));
                    }
                } catch (e) {
                    alert('Upload failed: ' + e.message);
                } finally {
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = 'Upload Selected Files';
                }
            });

            // Load stats
            async function loadStats() {
                try {
                    const resp = await fetch('/api/batch/stats');
                    const data = await resp.json();

                    document.getElementById('stat-pending').textContent = data.queue.pending;
                    document.getElementById('stat-processing').textContent = data.queue.processing;
                    document.getElementById('stat-completed').textContent = data.queue.completed;
                    document.getElementById('stat-failed').textContent = data.queue.failed;
                    document.getElementById('stat-visitors').textContent = data.total_visitors_detected.toLocaleString();
                } catch (e) {
                    console.error('Failed to load stats:', e);
                }
            }

            // Load jobs
            async function loadJobs() {
                try {
                    const resp = await fetch('/api/batch/jobs?limit=50');
                    const data = await resp.json();

                    const tbody = document.getElementById('jobs-tbody');

                    if (data.jobs.length === 0) {
                        tbody.innerHTML = `
                            <tr>
                                <td colspan="7">
                                    <div class="empty-state">
                                        <div class="icon">📭</div>
                                        <div>No jobs yet. Upload some videos to get started!</div>
                                    </div>
                                </td>
                            </tr>
                        `;
                        return;
                    }

                    tbody.innerHTML = data.jobs.map(job => `
                        <tr data-job-id="${job.id}">
                            <td title="${job.video_name}">${truncate(job.video_name, 30)}</td>
                            <td><a href="/analytics-dashboard/${job.venue_id}">${job.venue_id}</a></td>
                            <td><span class="status-badge status-${job.status}">${job.status}</span></td>
                            <td>
                                ${job.status === 'processing' ? `
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: ${job.progress}%"></div>
                                    </div>
                                ` : job.status === 'completed' ? '100%' : '-'}
                            </td>
                            <td>${job.visitors_detected || '-'}</td>
                            <td>${formatTime(job.created_at)}</td>
                            <td>
                                ${job.status === 'pending' ? `
                                    <button class="btn btn-small btn-danger" onclick="cancelJob('${job.id}')">Cancel</button>
                                ` : job.status === 'failed' ? `
                                    <span title="${job.error_message || 'Unknown error'}">⚠️</span>
                                ` : ''}
                            </td>
                        </tr>
                    `).join('');
                } catch (e) {
                    console.error('Failed to load jobs:', e);
                }
            }

            function truncate(str, len) {
                if (!str) return '-';
                return str.length > len ? str.slice(0, len) + '...' : str;
            }

            function formatTime(iso) {
                if (!iso) return '-';
                const d = new Date(iso);
                return d.toLocaleString();
            }

            async function cancelJob(jobId) {
                if (!confirm('Cancel this job?')) return;

                try {
                    const resp = await fetch(`/api/batch/jobs/${jobId}`, { method: 'DELETE' });
                    if (resp.ok) {
                        loadJobs();
                        loadStats();
                    } else {
                        const data = await resp.json();
                        alert('Failed to cancel: ' + (data.detail || 'Unknown error'));
                    }
                } catch (e) {
                    alert('Failed to cancel: ' + e.message);
                }
            }

            // Initial load and auto-refresh
            loadStats();
            loadJobs();
            setInterval(() => {
                loadStats();
                loadJobs();
            }, 3000);  // Refresh every 3 seconds
        </script>
    </body>
    </html>
    """


# =============================================================================
# MAP VISUALIZATION (Phase 3)
# =============================================================================

@app.get("/map", response_class=HTMLResponse)
async def map_view():
    """Interactive map showing all venues with analytics."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Venue Map - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
            }
            /* Navigation */
            nav {
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1001;
            }
            nav .nav-inner {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }
            nav .logo {
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }
            nav .logo span { color: #3b82f6; }
            nav .nav-links { display: flex; gap: 30px; }
            nav .nav-links a {
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }
            nav .nav-links a:hover { color: #fff; }
            nav .nav-links a.active { color: #3b82f6; }
            #map { width: 100%; height: calc(100vh - 60px); margin-top: 60px; }
            .legend {
                position: absolute;
                bottom: 30px;
                right: 10px;
                z-index: 1000;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 8px;
            }
            .legend h4 { margin-bottom: 10px; color: #fff; font-size: 14px; }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
                margin: 5px 0;
                font-size: 12px;
            }
            .legend-color {
                width: 16px;
                height: 16px;
                border-radius: 50%;
            }
            .venue-popup h3 { margin: 0 0 10px 0; color: #333; }
            .venue-popup p { margin: 5px 0; color: #666; font-size: 13px; }
            .venue-popup .stat { font-weight: bold; color: #0066ff; }
            .venue-popup a {
                display: inline-block;
                margin-top: 10px;
                padding: 8px 16px;
                background: #0066ff;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-size: 12px;
            }
            .stats-panel {
                position: absolute;
                top: 80px;
                left: 10px;
                z-index: 1000;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 8px;
                min-width: 200px;
            }
            .stats-panel h4 { margin-bottom: 10px; color: #fff; }
            .stats-panel .stat-row {
                display: flex;
                justify-content: space-between;
                margin: 8px 0;
                font-size: 13px;
            }
            .stats-panel .stat-value { color: #3b82f6; font-weight: bold; }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map" class="active">Map</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div id="map"></div>

        <div class="stats-panel" id="stats-panel">
            <h4>🗺️ Overview</h4>
            <div class="stat-row">
                <span>Total Venues</span>
                <span class="stat-value" id="total-venues">-</span>
            </div>
            <div class="stat-row">
                <span>Total Visitors</span>
                <span class="stat-value" id="total-visitors">-</span>
            </div>
            <div class="stat-row">
                <span>Countries</span>
                <span class="stat-value" id="total-countries">-</span>
            </div>
        </div>

        <div class="legend">
            <h4>Traffic Level</h4>
            <div class="legend-item">
                <div class="legend-color" style="background: #22c55e;"></div>
                <span>High (50+ visitors)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #eab308;"></div>
                <span>Medium (10-50)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #3b82f6;"></div>
                <span>Low (< 10)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #6b7280;"></div>
                <span>No data</span>
            </div>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            // Initialize map centered on Africa with better styling
            const map = L.map('map', {
                zoomControl: false  // We'll add custom position
            }).setView([-1.2921, 20.0], 3);

            // Add zoom control to bottom right
            L.control.zoom({ position: 'bottomleft' }).addTo(map);

            // Use Stadia Alidade Smooth Dark (free, beautiful)
            L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png', {
                maxZoom: 20,
                attribution: '&copy; Stadia Maps, &copy; OpenMapTiles, &copy; OpenStreetMap'
            }).addTo(map);

            // Fetch venues and analytics
            async function loadVenues() {
                try {
                    const response = await fetch('/api/map/venues');
                    const data = await response.json();

                    // Update stats
                    document.getElementById('total-venues').textContent = data.venues.length;
                    document.getElementById('total-visitors').textContent =
                        data.venues.reduce((sum, v) => sum + (v.visitors || 0), 0).toLocaleString();

                    const countries = new Set(data.venues.map(v => v.country).filter(c => c));
                    document.getElementById('total-countries').textContent = countries.size || '-';

                    // Add markers with pulse animation for high traffic
                    data.venues.forEach(venue => {
                        if (venue.latitude && venue.longitude) {
                            const color = getMarkerColor(venue.visitors);
                            const size = Math.max(10, Math.min(25, (venue.visitors || 0) / 3 + 10));

                            // Create custom icon with glow effect
                            const marker = L.circleMarker([venue.latitude, venue.longitude], {
                                radius: size,
                                fillColor: color,
                                color: color,
                                weight: 3,
                                opacity: 0.3,
                                fillOpacity: 0.9
                            }).addTo(map);

                            // Add inner dot
                            L.circleMarker([venue.latitude, venue.longitude], {
                                radius: size * 0.4,
                                fillColor: '#fff',
                                color: '#fff',
                                weight: 0,
                                fillOpacity: 0.9
                            }).addTo(map);

                            const venueTypeIcon = {
                                'bar': '🍺', 'restaurant': '🍽️', 'cafe': '☕',
                                'retail': '🛍️', 'nightclub': '🎵', 'hotel': '🏨',
                                'mall': '🏬', 'other': '📍'
                            }[venue.venue_type] || '📍';

                            marker.bindPopup(`
                                <div class="venue-popup">
                                    <h3>${venueTypeIcon} ${venue.name || venue.id}</h3>
                                    <p><strong>Type:</strong> ${venue.venue_type || 'Unknown'}</p>
                                    <p><strong>Location:</strong> ${[venue.city, venue.country].filter(x=>x).join(', ') || 'Not set'}</p>
                                    <hr style="border: none; border-top: 1px solid #eee; margin: 10px 0;">
                                    <p><strong>Total Visitors:</strong> <span class="stat">${(venue.visitors || 0).toLocaleString()}</span></p>
                                    <p><strong>Zone ID:</strong> <code style="background:#f0f0f0;padding:2px 6px;border-radius:3px;font-size:11px;">${venue.h3_zone || 'N/A'}</code></p>
                                    <a href="/analytics-dashboard/${venue.id}">View Full Analytics →</a>
                                </div>
                            `, { maxWidth: 300 });
                        }
                    });

                    // Fit bounds if we have venues
                    const venuesWithLocation = data.venues.filter(v => v.latitude && v.longitude);
                    if (venuesWithLocation.length > 0) {
                        const bounds = L.latLngBounds(
                            venuesWithLocation.map(v => [v.latitude, v.longitude])
                        );
                        map.fitBounds(bounds, { padding: [50, 50] });
                    }

                } catch (e) {
                    console.error('Failed to load venues:', e);
                }
            }

            function getMarkerColor(visitors) {
                if (!visitors) return '#6b7280';  // Gray - no data
                if (visitors >= 50) return '#22c55e';  // Green - high
                if (visitors >= 10) return '#eab308';  // Yellow - medium
                return '#3b82f6';  // Blue - low
            }

            loadVenues();
        </script>
    </body>
    </html>
    """


@app.get("/api/map/venues")
async def get_map_venues():
    """Get all venues with their analytics for map display."""
    # Get all venues with location
    venue_query = sqlalchemy.select(
        venues.c.id, venues.c.name, venues.c.latitude, venues.c.longitude,
        venues.c.h3_zone, venues.c.city, venues.c.country, venues.c.venue_type
    )
    venue_rows = await database.fetch_all(venue_query)

    result = []
    for v in venue_rows:
        # Get visitor count for this venue
        count_query = sqlalchemy.select(
            func.count(func.distinct(events.c.pseudo_id))
        ).where(events.c.venue_id == v["id"])
        visitors = await database.fetch_val(count_query) or 0

        result.append({
            "id": v["id"],
            "name": v["name"],
            "latitude": v["latitude"],
            "longitude": v["longitude"],
            "h3_zone": v["h3_zone"],
            "city": v["city"],
            "country": v["country"],
            "venue_type": v["venue_type"],
            "visitors": visitors
        })

    return {"venues": result}


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
