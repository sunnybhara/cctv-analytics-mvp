"""
Batch Processing Queue
======================
Background worker that processes the video job queue.
"""

import os
import threading
import asyncio
from datetime import datetime

import sqlalchemy
import databases

from app.config import DATABASE_URL
from app.database import jobs
from app.state import processing_jobs, _queue_lock, _queue_processor_running
from app.video.pipeline import process_video_file


async def create_job_in_db(job_id: str, venue_id: str, video_source: str, video_name: str, priority: int = 0):
    """Create a job record in the database."""
    from app.database import database
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


async def update_job_status(job_id: str, **kwargs):
    """Update job status in database."""
    from app.database import database
    query = jobs.update().where(jobs.c.id == job_id).values(**kwargs)
    await database.execute(query)


def _process_queue_sync():
    """Background worker that processes the job queue (runs in thread)."""
    import app.state as state

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    db = databases.Database(DATABASE_URL)
    loop.run_until_complete(db.connect())

    try:
        while True:
            query = sqlalchemy.select(jobs).where(
                jobs.c.status == "pending"
            ).order_by(
                jobs.c.priority.desc(),
                jobs.c.created_at.asc()
            ).limit(1)

            job = loop.run_until_complete(db.fetch_one(query))

            if not job:
                with state._queue_lock:
                    state._queue_processor_running = False
                break

            job_id = job["id"]
            venue_id = job["venue_id"]
            video_source = job["video_source"]

            update_query = jobs.update().where(jobs.c.id == job_id).values(
                status="processing",
                started_at=datetime.utcnow()
            )
            loop.run_until_complete(db.execute(update_query))

            processing_jobs[job_id] = {
                "status": "processing",
                "message": "Processing video...",
                "venue_id": venue_id,
                "current_frame": 0,
                "total_frames": 0
            }

            try:
                process_video_file(job_id, video_source, venue_id, DATABASE_URL)

                final_status = processing_jobs.get(job_id, {})
                visitors = final_status.get("visitors_detected", 0)

                update_query = jobs.update().where(jobs.c.id == job_id).values(
                    status="completed",
                    completed_at=datetime.utcnow(),
                    progress=100,
                    visitors_detected=visitors
                )
                loop.run_until_complete(db.execute(update_query))

            except Exception as e:
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

            if os.path.exists(video_source):
                try:
                    os.remove(video_source)
                    parent = os.path.dirname(video_source)
                    if parent and os.path.isdir(parent) and not os.listdir(parent):
                        os.rmdir(parent)
                except Exception:
                    pass

    finally:
        loop.run_until_complete(db.disconnect())
        loop.close()


def ensure_queue_processor_running():
    """Start the queue processor if not already running."""
    import app.state as state

    with state._queue_lock:
        if not state._queue_processor_running:
            state._queue_processor_running = True
            thread = threading.Thread(target=_process_queue_sync, daemon=True)
            thread.start()
