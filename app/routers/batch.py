"""
Batch Processing Endpoints (Phase 4)
=====================================
Upload multiple videos, queue YouTube URLs, list/cancel jobs, and queue stats.
"""

import os
import uuid
import tempfile
from datetime import datetime
from typing import Optional, List
from urllib.parse import urlparse

import sqlalchemy
from sqlalchemy import func
from fastapi import Depends, APIRouter, HTTPException, UploadFile, File, Form, Request
from app.auth import require_api_key
from app import limiter

from app.config import ALLOWED_VIDEO_DOMAINS, MAX_UPLOAD_SIZE_BYTES
from app.database import database, jobs
from app.state import processing_jobs
from app.video.deps import load_video_deps
from app.video.queue import create_job_in_db, ensure_queue_processor_running

router = APIRouter()


@limiter.limit("10/minute")
@router.post("/api/batch/upload")
async def batch_upload(
    request: Request,
    files: List[UploadFile] = File(...),
    venue_id: str = Form(default="demo_venue"),
    priority: int = Form(default=0),
    _api_key: str = Depends(require_api_key)
):
    """
    Upload multiple videos for batch processing.

    Returns list of job IDs for tracking progress.
    """
    # Early content-length check
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB")

    created_jobs = []

    for file in files:
        job_id = str(uuid.uuid4())

        # Save uploaded file with streaming size check
        temp_dir = tempfile.mkdtemp()
        safe_name = os.path.basename(file.filename or "video.mp4")
        temp_path = os.path.join(temp_dir, safe_name)

        bytes_written = 0
        with open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_SIZE_BYTES:
                    f.close()
                    os.remove(temp_path)
                    os.rmdir(temp_dir)
                    raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB")
                f.write(chunk)

        # Create job in database
        await create_job_in_db(
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
    ensure_queue_processor_running()

    return {
        "message": f"Queued {len(created_jobs)} videos for processing",
        "jobs": created_jobs
    }

@limiter.limit("10/minute")
@router.post("/api/batch/url")
async def batch_url(
    request: Request,
    urls: List[str],
    venue_id: str = "demo_venue",
    priority: int = 0,
    _api_key: str = Depends(require_api_key)
):
    """
    Queue multiple YouTube URLs for batch processing.
    """
    load_video_deps()
    from app.video.deps import yt_dlp

    created_jobs = []

    for url in urls:
        # Validate URL domain to prevent SSRF
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            if hostname not in ALLOWED_VIDEO_DOMAINS:
                created_jobs.append({
                    "url": url,
                    "status": "failed",
                    "error": f"Only YouTube URLs are supported. Allowed domains: {', '.join(ALLOWED_VIDEO_DOMAINS)}"
                })
                continue
        except Exception:
            created_jobs.append({
                "url": url,
                "status": "failed",
                "error": "Invalid URL format"
            })
            continue
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
        await create_job_in_db(
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
    ensure_queue_processor_running()

    return {
        "message": f"Queued {len([j for j in created_jobs if j['status'] == 'pending'])} videos for processing",
        "jobs": created_jobs
    }

@router.get("/api/batch/jobs")
async def list_jobs(
    venue_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    _api_key: str = Depends(require_api_key)
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

@router.get("/api/batch/jobs/{job_id}")
async def get_job(job_id: str, _api_key: str = Depends(require_api_key)):
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

@router.delete("/api/batch/jobs/{job_id}")
async def cancel_job(job_id: str, _api_key: str = Depends(require_api_key)):
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
        except Exception:
            pass

    return {"message": "Job cancelled", "job_id": job_id}

@router.get("/api/batch/stats")
async def batch_stats(_api_key: str = Depends(require_api_key)):
    """Get queue statistics."""
    from app.state import _queue_processor_running

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
