"""
Batch Processing Queue
======================
Parallel background worker pool that processes the video job queue.
Uses ThreadPoolExecutor with configurable MAX_PARALLEL_WORKERS.
"""

import os
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime

import sqlalchemy
import databases

from app.config import DATABASE_URL, MAX_PARALLEL_WORKERS
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


def _cleanup_video(video_source):
    """Remove processed video file and empty parent directory."""
    if os.path.exists(video_source):
        try:
            os.remove(video_source)
            parent = os.path.dirname(video_source)
            if parent and os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except Exception:
            pass


def _process_queue_sync():
    """Parallel background worker pool for video processing.

    Runs up to MAX_PARALLEL_WORKERS videos simultaneously using ThreadPoolExecutor.
    Each worker gets its own YOLO model instance (tracker state isolation).
    """
    import app.state as state

    if MAX_PARALLEL_WORKERS > 2:
        print(f"Warning: MAX_PARALLEL_WORKERS={MAX_PARALLEL_WORKERS}. "
              f"Each worker uses ~200MB RAM for YOLO. Monitor memory usage.")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    db = databases.Database(DATABASE_URL)
    loop.run_until_complete(db.connect())

    try:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            futures = {}  # future -> job dict

            while True:
                # Fill available worker slots with pending jobs
                available_slots = MAX_PARALLEL_WORKERS - len(futures)
                if available_slots > 0:
                    query = sqlalchemy.select(jobs).where(
                        jobs.c.status == "pending"
                    ).order_by(
                        jobs.c.priority.desc(),
                        jobs.c.created_at.asc()
                    ).limit(available_slots)

                    pending_jobs = loop.run_until_complete(db.fetch_all(query))

                    for job in pending_jobs:
                        job_id = job["id"]
                        venue_id = job["venue_id"]
                        video_source = job["video_source"]

                        # Mark as processing in DB
                        loop.run_until_complete(db.execute(
                            jobs.update().where(jobs.c.id == job_id).values(
                                status="processing",
                                started_at=datetime.utcnow()
                            )
                        ))

                        processing_jobs[job_id] = {
                            "status": "processing",
                            "message": "Processing video...",
                            "venue_id": venue_id,
                            "current_frame": 0,
                            "total_frames": 0
                        }

                        future = executor.submit(
                            process_video_file, job_id, video_source, venue_id, DATABASE_URL
                        )
                        futures[future] = {
                            "id": job_id,
                            "venue_id": venue_id,
                            "video_source": video_source,
                        }

                if not futures:
                    # No running jobs and no pending jobs — exit
                    break

                # Wait for at least one worker to finish (5s timeout to re-check for new jobs)
                done, _ = wait(futures.keys(), timeout=5, return_when=FIRST_COMPLETED)

                for future in done:
                    job_info = futures.pop(future)
                    job_id = job_info["id"]

                    try:
                        future.result()  # Raises if worker threw an exception

                        final_status = processing_jobs.get(job_id, {})
                        visitors = final_status.get("visitors_detected", 0)

                        loop.run_until_complete(db.execute(
                            jobs.update().where(jobs.c.id == job_id).values(
                                status="completed",
                                completed_at=datetime.utcnow(),
                                progress=100,
                                visitors_detected=visitors
                            )
                        ))

                    except Exception as e:
                        loop.run_until_complete(db.execute(
                            jobs.update().where(jobs.c.id == job_id).values(
                                status="failed",
                                completed_at=datetime.utcnow(),
                                error_message=str(e)
                            )
                        ))

                        processing_jobs[job_id] = {
                            "status": "failed",
                            "message": f"Error: {str(e)}",
                            "venue_id": job_info["venue_id"]
                        }

                    _cleanup_video(job_info["video_source"])

    finally:
        # ALWAYS reset the flag — even on unhandled exceptions.
        with state._queue_lock:
            state._queue_processor_running = False

        loop.run_until_complete(db.disconnect())
        loop.close()

        # Re-check: if new jobs arrived while shutting down, restart.
        try:
            check_loop = asyncio.new_event_loop()
            check_db = databases.Database(DATABASE_URL)
            check_loop.run_until_complete(check_db.connect())
            pending = check_loop.run_until_complete(check_db.fetch_one(
                sqlalchemy.select(jobs.c.id).where(jobs.c.status == "pending").limit(1)
            ))
            check_loop.run_until_complete(check_db.disconnect())
            check_loop.close()
            if pending:
                ensure_queue_processor_running()
        except Exception:
            pass


def ensure_queue_processor_running():
    """Start the queue processor if not already running."""
    import app.state as state

    with state._queue_lock:
        if not state._queue_processor_running:
            state._queue_processor_running = True
            thread = threading.Thread(target=_process_queue_sync, daemon=True)
            thread.start()
