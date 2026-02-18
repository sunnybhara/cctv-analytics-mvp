"""
Visitor Embedding Database Operations
======================================
Synchronous DB operations for visitor embeddings, used by worker threads.
"""

from datetime import datetime, timedelta

import sqlalchemy

from app.config import DATABASE_URL, EMBEDDING_RETENTION_DAYS
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
    """Load non-expired embeddings for a venue (synchronous, for use in threads).

    Filters out embeddings older than EMBEDDING_RETENTION_DAYS as defense-in-depth
    (expired embeddings may still exist between purge cycles).
    """
    engine = _get_sync_engine(db_url)
    embeddings = []
    query = sqlalchemy.select(visitor_embeddings).where(
        visitor_embeddings.c.venue_id == venue_id
    )
    # Filter expired embeddings on load
    if EMBEDDING_RETENTION_DAYS > 0:
        cutoff = datetime.utcnow() - timedelta(days=EMBEDDING_RETENTION_DAYS)
        query = query.where(visitor_embeddings.c.last_seen >= cutoff)
    with engine.connect() as conn:
        result = conn.execute(query)
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


def purge_expired_embeddings_sync(db_url: str = None, retention_days: int = None, venue_id: str = None):
    """Delete embeddings older than retention_days.

    Args:
        db_url: Database URL (uses default if None)
        retention_days: Override for EMBEDDING_RETENTION_DAYS (uses config if None)
        venue_id: If set, only purge for this venue

    Returns:
        Number of deleted rows
    """
    days = retention_days if retention_days is not None else EMBEDDING_RETENTION_DAYS
    if days <= 0:
        return 0

    cutoff = datetime.utcnow() - timedelta(days=days)
    engine = _get_sync_engine(db_url)

    condition = visitor_embeddings.c.last_seen < cutoff
    if venue_id:
        condition = condition & (visitor_embeddings.c.venue_id == venue_id)

    with engine.connect() as conn:
        result = conn.execute(visitor_embeddings.delete().where(condition))
        conn.commit()
        return result.rowcount
