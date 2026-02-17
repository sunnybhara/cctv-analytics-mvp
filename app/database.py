"""
Database Setup
==============
SQLAlchemy table definitions, async database connection, and engine.
All 5 tables defined here as module-level objects importable by routers and workers.
"""

from datetime import datetime

import databases
import sqlalchemy
from sqlalchemy import func

from app.config import DATABASE_URL as _RAW_DB_URL

# Force psycopg3 dialect for Python 3.13 compatibility
DATABASE_URL = _RAW_DB_URL
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

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
    sqlalchemy.Column("track_frames", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("detection_conf", sqlalchemy.Float, default=0.0),
    # Behavior/Engagement fields
    sqlalchemy.Column("engagement_score", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("behavior_type", sqlalchemy.String(20), nullable=True),
    sqlalchemy.Column("body_orientation", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("posture", sqlalchemy.String(20), nullable=True),
)

# Venues table with geo-location
venues = sqlalchemy.Table(
    "venues",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String(50), primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(100)),
    sqlalchemy.Column("api_key", sqlalchemy.String(64)),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("config", sqlalchemy.JSON, default=dict),
    sqlalchemy.Column("latitude", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("longitude", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("h3_zone", sqlalchemy.String(20), nullable=True, index=True),
    sqlalchemy.Column("address", sqlalchemy.String(500), nullable=True),
    sqlalchemy.Column("city", sqlalchemy.String(100), nullable=True),
    sqlalchemy.Column("country", sqlalchemy.String(100), nullable=True),
    sqlalchemy.Column("venue_type", sqlalchemy.String(50), nullable=True),
)

# Jobs table - tracks video processing queue
jobs = sqlalchemy.Table(
    "jobs",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String(36), primary_key=True),
    sqlalchemy.Column("venue_id", sqlalchemy.String(50), index=True),
    sqlalchemy.Column("status", sqlalchemy.String(20), index=True, default="pending"),
    sqlalchemy.Column("video_source", sqlalchemy.String(500)),
    sqlalchemy.Column("video_name", sqlalchemy.String(200)),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("started_at", sqlalchemy.DateTime, nullable=True),
    sqlalchemy.Column("completed_at", sqlalchemy.DateTime, nullable=True),
    sqlalchemy.Column("progress", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("frames_processed", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("total_frames", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("visitors_detected", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("error_message", sqlalchemy.Text, nullable=True),
    sqlalchemy.Column("priority", sqlalchemy.Integer, default=0),
)

# Alerts table - stores anomalies and notifications
alerts = sqlalchemy.Table(
    "alerts",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("venue_id", sqlalchemy.String(50), index=True),
    sqlalchemy.Column("alert_type", sqlalchemy.String(50)),
    sqlalchemy.Column("severity", sqlalchemy.String(20), default="info"),
    sqlalchemy.Column("title", sqlalchemy.String(200)),
    sqlalchemy.Column("message", sqlalchemy.Text),
    sqlalchemy.Column("data", sqlalchemy.JSON, nullable=True),
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
    sqlalchemy.Column("visitor_id", sqlalchemy.String(32), unique=True, index=True),
    sqlalchemy.Column("embedding", sqlalchemy.LargeBinary),
    sqlalchemy.Column("embedding_model", sqlalchemy.String(50), default="arcface"),
    sqlalchemy.Column("first_seen", sqlalchemy.DateTime, index=True),
    sqlalchemy.Column("last_seen", sqlalchemy.DateTime, index=True),
    sqlalchemy.Column("visit_count", sqlalchemy.Integer, default=1),
    sqlalchemy.Column("total_dwell_seconds", sqlalchemy.Float, default=0),
    sqlalchemy.Column("age_bracket", sqlalchemy.String(10), nullable=True),
    sqlalchemy.Column("gender", sqlalchemy.String(1), nullable=True),
    sqlalchemy.Column("quality_score", sqlalchemy.Float, default=0),
)

# Cohorts table - groups of venues for comparative analytics
cohorts = sqlalchemy.Table(
    "cohorts",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String(50), primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(100)),
    sqlalchemy.Column("color", sqlalchemy.String(7), default="#3b82f6"),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
)

# Cohort membership - many-to-many link between venues and cohorts
cohort_members = sqlalchemy.Table(
    "cohort_members",
    metadata,
    sqlalchemy.Column("cohort_id", sqlalchemy.String(50), index=True),
    sqlalchemy.Column("venue_id", sqlalchemy.String(50), index=True),
    sqlalchemy.UniqueConstraint("cohort_id", "venue_id"),
)

engine = sqlalchemy.create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
metadata.create_all(engine)
