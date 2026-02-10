"""
Visitor Embedding Database Operations
======================================
Synchronous DB operations for visitor embeddings, used by worker threads.
"""

from datetime import datetime

import sqlalchemy

from app.config import DATABASE_URL
from app.database import visitor_embeddings

# Shared sync engine (created once, reused)
_sync_engine = None


def _get_sync_engine(db_url: str = None):
    """Get or create the synchronous database engine."""
    global _sync_engine
    url = db_url or DATABASE_URL
    if _sync_engine is None:
        _sync_engine = sqlalchemy.create_engine(
            url,
            connect_args={"check_same_thread": False} if "sqlite" in url else {}
        )
    return _sync_engine


def load_venue_embeddings_sync(venue_id: str, db_url: str = None):
    """Load all embeddings for a venue (synchronous, for use in threads)."""
    engine = _get_sync_engine(db_url)
    embeddings = []
    with engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.select(visitor_embeddings).where(
                visitor_embeddings.c.venue_id == venue_id
            )
        )
        for row in result:
            embeddings.append(dict(row._mapping))
    return embeddings


def save_visitor_embedding_sync(
    db_url: str = None,
    venue_id: str = "",
    visitor_id: str = "",
    embedding_bytes: bytes = b"",
    timestamp: datetime = None,
    age_bracket: str = None,
    gender: str = None,
    quality_score: float = 0.0
):
    """Save a new visitor embedding (synchronous)."""
    engine = _get_sync_engine(db_url)
    with engine.connect() as conn:
        conn.execute(
            visitor_embeddings.insert().values(
                venue_id=venue_id,
                visitor_id=visitor_id,
                embedding=embedding_bytes,
                embedding_model="arcface",
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
    db_url: str = None,
    visitor_id: str = "",
    timestamp: datetime = None,
    dwell_seconds: float = 0
):
    """Update existing visitor's last_seen and visit_count (synchronous)."""
    engine = _get_sync_engine(db_url)
    with engine.connect() as conn:
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
