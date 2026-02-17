"""
SIPP Bar Detection Pipeline — Entry Point
==========================================
Video source -> YOLO detection -> interaction trigger -> Claude verification -> events
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import cv2

from config.settings import CAMERA_ID, CLIP_SAVE_DIR, VIDEO_SOURCE, ZONES_FILE
from pipeline.clip_buffer import ClipBuffer
from pipeline.deduplicator import EventDeduplicator
from pipeline.detector import DualTracker
from pipeline.events import BarEvent, post_events
from pipeline.interactions import InteractionDetector
from pipeline.verifier import verify_event
from pipeline.zones import load_zones

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sipp-pipeline")

# Hardcoded for Week 1. Admin panel sets this in Sprint 2.
VENUE_ID = "demo_venue"

# Max concurrent VLM calls (prevents API overload while unblocking frame loop)
VLM_WORKER_THREADS = 2


@dataclass
class _PendingVLM:
    """Holds context needed to process a VLM result after it returns."""
    pending_event: object  # The interaction event that triggered VLM
    keyframes: list


def _run_vlm_sync(pending_vlm: _PendingVLM) -> tuple[_PendingVLM, dict]:
    """Run VLM verification in a background thread (blocking call)."""
    result = verify_event(
        pending_vlm.keyframes,
        {"object_class": pending_vlm.pending_event.object_class,
         "zone": pending_vlm.pending_event.zone},
    )
    return pending_vlm, result


async def run_pipeline():
    """Main pipeline loop. Process video frame by frame.

    VLM calls run in a ThreadPoolExecutor so the frame loop continues
    reading frames while waiting for Claude API responses.
    """

    logger.info(f"Starting pipeline: source={VIDEO_SOURCE}, camera={CAMERA_ID}")

    zones = load_zones(ZONES_FILE, CAMERA_ID)
    logger.info(f"Loaded {len(zones)} zones: {list(zones.keys())}")

    tracker = DualTracker()
    interaction_detector = InteractionDetector()
    clip_buffer = ClipBuffer()
    deduplicator = EventDeduplicator()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {VIDEO_SOURCE}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clip_buffer.fps = fps
    frame_count = 0
    events_batch: list[BarEvent] = []

    logger.info(f"Video opened: {fps:.1f} FPS")

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=VLM_WORKER_THREADS)
    vlm_futures: list[asyncio.Future] = []

    def _process_vlm_result(pending_vlm: _PendingVLM, vlm_result: dict):
        """Process a completed VLM result and create event if valid."""
        action = vlm_result.get("action", "unknown")
        confidence = vlm_result.get("confidence", 0.0)

        # Guard against non-numeric confidence from VLM
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        if action in ("unknown", "idle", "cleaning") or confidence < 0.5:
            logger.debug(f"Skipped: {action} ({confidence:.2f})")
            return

        pending = pending_vlm.pending_event
        if not deduplicator.should_emit(pending.zone, action, confidence):
            logger.debug(f"Deduped: {action} in {pending.zone}")
            return

        if "pour" in action or "serv" in action:
            event_type = "serve"
        elif "payment" in action:
            event_type = "payment"
        else:
            event_type = "activity"

        event = BarEvent(
            event_type=event_type,
            venue_id=VENUE_ID,
            camera_id=CAMERA_ID,
            zone=pending.zone,
            confidence=confidence,
            action=action,
            drink_category=vlm_result.get("drink_category", "unknown"),
            vessel_type=vlm_result.get("vessel_type", "unknown"),
            description=vlm_result.get("description", ""),
            person_track_id=pending.person_track_id,
            object_track_id=pending.object_track_id,
        )

        events_batch.append(event)

        clip_path = clip_buffer.save_clip(pending.timestamp, event.event_id)
        if clip_path:
            logger.info(f"Saved training clip: {clip_path}")

        logger.info(
            f"EVENT: {action} | {vlm_result.get('drink_category')} | "
            f"conf={confidence:.2f} | zone={pending.zone}"
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            frame_count += 1

            clip_buffer.add_frame(frame, now)

            persons, objects = tracker.process_frame(frame, zones)

            pending_events = interaction_detector.detect(persons, objects)

            # Submit VLM calls to thread pool (non-blocking)
            for pending in pending_events:
                keyframes = clip_buffer.extract_clip_frames(pending.timestamp)
                if not keyframes:
                    continue

                pending_vlm = _PendingVLM(pending_event=pending, keyframes=keyframes)
                future = loop.run_in_executor(executor, _run_vlm_sync, pending_vlm)
                vlm_futures.append(future)

            # Collect completed VLM results (non-blocking check)
            still_pending = []
            for future in vlm_futures:
                if future.done():
                    try:
                        pending_vlm, vlm_result = future.result()
                        _process_vlm_result(pending_vlm, vlm_result)
                    except Exception as e:
                        logger.error(f"VLM worker error: {e}")
                else:
                    still_pending.append(future)
            vlm_futures = still_pending

            # Batch post events every 10 seconds
            if events_batch and frame_count % max(1, int(fps * 10)) == 0:
                success = await post_events(events_batch)
                if success:
                    events_batch = []

            # Log progress every 30 seconds
            if frame_count % max(1, int(fps * 30)) == 0:
                logger.info(
                    f"Processed {frame_count} frames, "
                    f"{len(persons)} persons, {len(objects)} objects in view, "
                    f"{len(vlm_futures)} VLM calls pending"
                )

        # Wait for remaining VLM futures before flushing
        if vlm_futures:
            logger.info(f"Waiting for {len(vlm_futures)} pending VLM calls...")
            done = await asyncio.gather(*vlm_futures, return_exceptions=True)
            for result in done:
                if isinstance(result, Exception):
                    logger.error(f"VLM worker error: {result}")
                else:
                    pending_vlm, vlm_result = result
                    _process_vlm_result(pending_vlm, vlm_result)

        # Flush remaining events
        if events_batch:
            await post_events(events_batch)

    finally:
        cap.release()
        executor.shutdown(wait=False)

    logger.info(f"Pipeline complete: {frame_count} frames processed")


if __name__ == "__main__":
    if not VIDEO_SOURCE:
        print("Set VIDEO_SOURCE env var to a file path or rtsp:// URL")
        exit(1)
    asyncio.run(run_pipeline())
