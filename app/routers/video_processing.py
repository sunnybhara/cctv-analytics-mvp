"""
Video Processing Endpoints
==========================
YouTube URL processing, file upload, and job status.
"""

import os
import uuid
import tempfile
import threading
import secrets
from urllib.parse import urlparse

import sqlalchemy
from fastapi import Depends, APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from app.auth import require_api_key
from app.responses import success_response
from app import limiter

from app.config import DATABASE_URL, ALLOWED_VIDEO_DOMAINS, MAX_UPLOAD_SIZE_BYTES
from app.database import database, venues
from app.state import processing_jobs
from app.video.deps import load_video_deps
from app.video.pipeline import process_video_file
from app.video.download import download_youtube_video
from app.video.helpers import lat_long_to_h3

router = APIRouter()


@limiter.limit("10/minute")
@router.post("/process/youtube")
async def process_youtube(request: Request, data: dict, background_tasks: BackgroundTasks, _api_key: str = Depends(require_api_key)):
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

    # Validate URL domain to prevent SSRF
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if hostname not in ALLOWED_VIDEO_DOMAINS:
            raise HTTPException(
                status_code=400,
                detail=f"Only YouTube URLs are supported. Allowed domains: {', '.join(ALLOWED_VIDEO_DOMAINS)}"
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid URL format")

    # Ensure venue row always exists (so delete can find it later)
    h3_zone = None
    if latitude is not None and longitude is not None:
        h3_zone = lat_long_to_h3(latitude, longitude)

    existing = await database.fetch_one(
        sqlalchemy.select(venues.c.id).where(venues.c.id == venue_id)
    )

    if existing:
        # Update existing venue with any new data
        update_vals = {}
        if venue_name:
            update_vals["name"] = venue_name
        if latitude is not None:
            update_vals["latitude"] = latitude
            update_vals["longitude"] = longitude
            update_vals["h3_zone"] = h3_zone
        if city:
            update_vals["city"] = city
        if country:
            update_vals["country"] = country
        if venue_type:
            update_vals["venue_type"] = venue_type
        if update_vals:
            await database.execute(
                venues.update().where(venues.c.id == venue_id).values(**update_vals)
            )
    else:
        # Always create venue row so data is never orphaned
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
            except Exception:
                pass

        except Exception as e:
            processing_jobs[job_id]["status"] = "error"
            processing_jobs[job_id]["message"] = str(e)

    thread = threading.Thread(target=download_and_process)
    thread.start()

    return success_response({"job_id": job_id, "status": "started"})


@limiter.limit("10/minute")
@router.post("/process/upload")
async def process_upload(
    request: Request,
    file: UploadFile = File(...),
    venue_id: str = Form(default="demo_venue"),
    _api_key: str = Depends(require_api_key)
):
    """Process an uploaded video file."""
    # Early content-length check
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB")

    job_id = str(uuid.uuid4())[:8]

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
            except Exception:
                pass

    thread = threading.Thread(target=process_and_cleanup)
    thread.start()

    return success_response({"job_id": job_id, "status": "started"})


@router.get("/process/status/{job_id}")
async def get_process_status(job_id: str, _api_key: str = Depends(require_api_key)):
    """Get the status of a processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return success_response(processing_jobs[job_id])
