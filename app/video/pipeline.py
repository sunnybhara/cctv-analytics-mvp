"""
Video Processing Pipeline
=========================
Core video processing: YOLO11 detection, BoT-SORT tracking, demographics, ReID, behavior analysis.
"""

import os
import traceback
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, Any

from app.state import processing_jobs
from app.database import events, visitor_embeddings
from app.config import DATABASE_URL
from app.video.deps import load_video_deps, cv2, reid_module, behavior_module
from app.video.models import get_yolo_model
from app.video.helpers import generate_pseudo_id, get_zone, estimate_demographics_from_crop
from app.video.embeddings import load_venue_embeddings_sync, save_visitor_embedding_sync, update_visitor_embedding_sync

# Path to BoT-SORT config (relative to project root)
_BOTSORT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "botsort.yaml")


def process_video_file(job_id: str, video_path: str, venue_id: str, db_url: str):
    """Process video file and generate events using YOLO11 + BoT-SORT."""
    global processing_jobs

    # Import fresh references to lazy-loaded modules
    from app.video.deps import cv2 as _cv2, reid_module as _reid, behavior_module as _behavior

    cap = None  # Initialize for cleanup in finally
    try:
        load_video_deps()

        # Re-import after loading deps to get the actual modules
        from app.video.deps import cv2 as _cv2, reid_module as _reid, behavior_module as _behavior

        processing_jobs[job_id]["status"] = "loading_model"
        processing_jobs[job_id]["message"] = "Loading YOLO11 model..."

        # Use pre-loaded YOLO model (or load on demand if not ready)
        model = get_yolo_model()

        # Initialize ReID for return visitor tracking
        reid_matcher = None
        reid_enabled = False
        if _reid is not None:
            try:
                processing_jobs[job_id]["message"] = "Loading return visitor data..."
                reid_matcher = _reid.VisitorMatcher(venue_id, similarity_threshold=0.68)

                # Load existing embeddings
                existing_embeddings = load_venue_embeddings_sync(venue_id, db_url)
                if existing_embeddings:
                    reid_matcher.load_embeddings(existing_embeddings)
                    processing_jobs[job_id]["known_visitors"] = reid_matcher.visitor_count
                    reid_enabled = True
                else:
                    reid_enabled = True
                    processing_jobs[job_id]["known_visitors"] = 0

                print(f"ReID enabled with {reid_matcher.visitor_count} known visitors")
            except Exception as e:
                print(f"ReID initialization failed: {e}")
                reid_matcher = None
                reid_enabled = False

        processing_jobs[job_id]["reid_enabled"] = reid_enabled

        processing_jobs[job_id]["status"] = "opening_video"
        processing_jobs[job_id]["message"] = "Opening video..."

        cap = _cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")

        fps = cap.get(_cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
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

        # Resolve BoT-SORT config path
        tracker_config = _BOTSORT_CONFIG if os.path.exists(_BOTSORT_CONFIG) else "botsort.yaml"

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

            # BoT-SORT tracking: detection + tracking in one call
            # persist=True maintains track IDs across frames
            results = model.track(
                frame,
                persist=True,
                tracker=tracker_config,
                classes=[0],    # persons only
                conf=0.5,
                verbose=False
            )

            # Process tracked detections
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                # Get track IDs (None if not yet assigned)
                track_ids = boxes.id
                if track_ids is None:
                    continue

                track_ids = track_ids.int().cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for i in range(len(track_ids)):
                    track_id = int(track_ids[i])
                    x1, y1, x2, y2 = xyxy[i]
                    conf = float(confs[i])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    h = y2 - y1

                    zone = get_zone(cx, cy, frame_width, frame_height)

                    if track_id not in track_data:
                        # New track - extract demographics, embedding, behavior
                        age, gender = None, None
                        embedding = None
                        embedding_quality = 0.0
                        person_crop = None

                        try:
                            px1 = max(0, int(x1))
                            py1 = max(0, int(y1))
                            px2 = min(frame_width, int(x2))
                            py2 = min(frame_height, int(y2))
                            person_crop = frame[py1:py2, px1:px2]

                            if person_crop.size > 0:
                                age, gender = estimate_demographics_from_crop(person_crop, h, frame_height)

                                # Try to extract face embedding for ReID
                                if reid_enabled and _reid is not None:
                                    try:
                                        embedding, embedding_quality = _reid.extract_face_embedding(person_crop)
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        # Analyze behavior if module available
                        behavior_result = None
                        if _behavior is not None and person_crop is not None and person_crop.size > 0:
                            try:
                                behavior_result = _behavior.analyze_behavior(person_crop)
                            except Exception:
                                pass

                        track_data[track_id] = {
                            "first_seen": current_time,
                            "last_seen": current_time,
                            "zones": [zone],
                            "age": age,
                            "gender": gender,
                            "events_created": False,
                            "frame_count": 1,
                            "conf_sum": conf,
                            "embedding": embedding,
                            "embedding_quality": embedding_quality,
                            "best_crop": person_crop.copy() if person_crop is not None and person_crop.size > 0 else None,
                            # Behavior data
                            "behavior_scores": [behavior_result.engagement_score] if behavior_result else [],
                            "behavior_types": [behavior_result.behavior_type] if behavior_result else [],
                            "body_orientations": [behavior_result.body_orientation] if behavior_result else [],
                            "postures": [behavior_result.posture] if behavior_result else [],
                            "prev_landmarks": behavior_result.landmarks if behavior_result else None,
                        }
                    else:
                        # Existing track - update
                        track_data[track_id]["last_seen"] = current_time
                        track_data[track_id]["frame_count"] += 1
                        track_data[track_id]["conf_sum"] += conf
                        if zone not in track_data[track_id]["zones"]:
                            track_data[track_id]["zones"].append(zone)

                        # Periodically update demographics, embedding, and behavior
                        needs_demographics = track_data[track_id]["age"] is None or track_data[track_id]["gender"] is None
                        needs_embedding = reid_enabled and track_data[track_id].get("embedding") is None
                        should_update_behavior = _behavior is not None and track_data[track_id]["frame_count"] % 3 == 0

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

                                    if needs_embedding and _reid is not None:
                                        try:
                                            embedding, quality = _reid.extract_face_embedding(person_crop)
                                            if embedding is not None and quality > track_data[track_id].get("embedding_quality", 0):
                                                track_data[track_id]["embedding"] = embedding
                                                track_data[track_id]["embedding_quality"] = quality
                                                track_data[track_id]["best_crop"] = person_crop.copy()
                                        except Exception:
                                            pass

                                    if _behavior is not None:
                                        try:
                                            prev_landmarks = track_data[track_id].get("prev_landmarks")
                                            behavior_result = _behavior.analyze_behavior(
                                                person_crop,
                                                previous_landmarks=prev_landmarks,
                                                time_delta=frame_interval / fps
                                            )
                                            track_data[track_id]["behavior_scores"].append(behavior_result.engagement_score)
                                            track_data[track_id]["behavior_types"].append(behavior_result.behavior_type)
                                            track_data[track_id]["body_orientations"].append(behavior_result.body_orientation)
                                            track_data[track_id]["postures"].append(behavior_result.posture)
                                            track_data[track_id]["prev_landmarks"] = behavior_result.landmarks
                                        except Exception:
                                            pass
                            except Exception:
                                pass

        processing_jobs[job_id]["status"] = "generating_events"
        processing_jobs[job_id]["message"] = "Generating events..."

        # Filter out short-lived tracks (noise/false detections)
        MIN_FRAMES = 8
        valid_tracks = {tid: data for tid, data in track_data.items()
                       if data.get("frame_count", 1) >= MIN_FRAMES}

        # DEDUPLICATION: Merge tracks that belong to the same person
        # Uses ArcFace face embeddings to detect when the tracker assigned
        # multiple IDs to the same person (e.g. after brief occlusion).
        MERGE_SIMILARITY = 0.60  # Lower than ReID threshold (0.68) to catch same-video duplicates
        def merge_duplicate_tracks(tracks):
            if len(tracks) <= 1 or _reid is None:
                return tracks

            # Separate tracks with and without embeddings
            with_emb = {tid: data for tid, data in tracks.items() if data.get("embedding") is not None}
            without_emb = {tid: data for tid, data in tracks.items() if data.get("embedding") is None}

            if len(with_emb) <= 1:
                return tracks

            # Build list for pairwise comparison, sorted by first_seen
            tids = sorted(with_emb.keys(), key=lambda t: with_emb[t]["first_seen"])

            # Union-find for merging
            parent = {tid: tid for tid in tids}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            # Compare all pairs — only merge if tracks DON'T overlap in time
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    if find(tids[i]) == find(tids[j]):
                        continue
                    a, b = with_emb[tids[i]], with_emb[tids[j]]
                    # Skip if tracks overlap (two different people visible at same time)
                    if a["first_seen"] <= b["last_seen"] and b["first_seen"] <= a["last_seen"]:
                        continue
                    sim = _reid.cosine_similarity(a["embedding"], b["embedding"])
                    if sim >= MERGE_SIMILARITY:
                        parent[find(tids[j])] = find(tids[i])

            # Group tracks by their root
            groups = {}
            for tid in tids:
                root = find(tid)
                groups.setdefault(root, []).append(tid)

            # Merge each group into a single track (keep the longest as primary)
            merged = {}
            for root, group_tids in groups.items():
                primary_tid = max(group_tids, key=lambda t: with_emb[t]["frame_count"])
                primary = dict(with_emb[primary_tid])

                for tid in group_tids:
                    if tid == primary_tid:
                        continue
                    other = with_emb[tid]
                    # Extend time range
                    if other["first_seen"] < primary["first_seen"]:
                        primary["first_seen"] = other["first_seen"]
                    if other["last_seen"] > primary["last_seen"]:
                        primary["last_seen"] = other["last_seen"]
                    # Merge zones
                    for z in other.get("zones", []):
                        if z not in primary["zones"]:
                            primary["zones"].append(z)
                    # Sum frames
                    primary["frame_count"] = primary.get("frame_count", 0) + other.get("frame_count", 0)
                    primary["conf_sum"] = primary.get("conf_sum", 0) + other.get("conf_sum", 0)
                    # Keep best embedding
                    if other.get("embedding_quality", 0) > primary.get("embedding_quality", 0):
                        primary["embedding"] = other["embedding"]
                        primary["embedding_quality"] = other["embedding_quality"]
                    # Merge behavior data
                    for key in ["behavior_scores", "behavior_types", "body_orientations", "postures"]:
                        primary.setdefault(key, []).extend(other.get(key, []))

                merged[primary_tid] = primary

            # Add back tracks without embeddings
            merged.update(without_emb)
            return merged

        valid_tracks = merge_duplicate_tracks(valid_tracks)

        # MAX-CONCURRENT CAP: If at most N people are ever visible at once,
        # there can be at most N unique people. Keep the N longest tracks.
        def cap_by_max_concurrent(tracks):
            if len(tracks) <= 1:
                return tracks

            # Sweep-line to find max simultaneous tracks
            time_events = []
            for tid, data in tracks.items():
                time_events.append((data["first_seen"], 0, tid))   # 0 = start (sort before end)
                time_events.append((data["last_seen"], 1, tid))    # 1 = end
            time_events.sort()

            active = set()
            max_concurrent = 0
            for ts, etype, tid in time_events:
                if etype == 0:
                    active.add(tid)
                else:
                    active.discard(tid)
                max_concurrent = max(max_concurrent, len(active))

            if len(tracks) <= max_concurrent:
                return tracks

            # Keep the top N tracks by frame_count (longest-lived = most confident)
            sorted_tids = sorted(tracks.keys(), key=lambda t: tracks[t].get("frame_count", 0), reverse=True)
            return {tid: tracks[tid] for tid in sorted_tids[:max_concurrent]}

        valid_tracks = cap_by_max_concurrent(valid_tracks)

        # Debug info
        processing_jobs[job_id]["debug_count_after"] = len(valid_tracks)
        processing_jobs[job_id]["message"] = f"Found {len(valid_tracks)} unique visitors (dedup + concurrent cap)..."

        # Generate events from valid track data with ReID for return visitors
        seen_pseudo_ids = set()
        new_visitors = []
        return_visitors = []
        reid_stats = {"matched": 0, "new": 0, "no_face": 0}

        for track_id, data in valid_tracks.items():
            dwell_seconds = (data["last_seen"] - data["first_seen"]).total_seconds()
            if dwell_seconds < 1:
                dwell_seconds = 1.0  # Minimum 1 second for very short tracks

            # Try ReID matching if we have an embedding
            visitor_id = None
            is_return_visitor = False
            reid_confidence = 0.0

            if reid_enabled and reid_matcher is not None and data.get("embedding") is not None:
                embedding = data["embedding"]

                matched_id, similarity = reid_matcher.find_match(embedding)

                if matched_id:
                    visitor_id = matched_id
                    is_return_visitor = True
                    reid_confidence = similarity
                    reid_stats["matched"] += 1

                    return_visitors.append({
                        "visitor_id": visitor_id,
                        "timestamp": data["last_seen"],
                        "dwell_seconds": dwell_seconds
                    })
                else:
                    visitor_id = _reid.generate_visitor_id(embedding)
                    reid_stats["new"] += 1

                    reid_matcher.add_visitor(visitor_id, embedding, {
                        "visitor_id": visitor_id,
                        "first_seen": data["first_seen"],
                        "last_seen": data["last_seen"],
                        "visit_count": 1,
                        "age_bracket": data["age"],
                        "gender": data["gender"]
                    })

                    new_visitors.append({
                        "visitor_id": visitor_id,
                        "embedding": embedding,
                        "embedding_quality": data.get("embedding_quality", 0),
                        "timestamp": data["first_seen"],
                        "age_bracket": data["age"],
                        "gender": data["gender"]
                    })
            else:
                visitor_id = generate_pseudo_id(track_id, date_str)
                reid_stats["no_face"] += 1

            is_repeat = visitor_id in seen_pseudo_ids or is_return_visitor
            seen_pseudo_ids.add(visitor_id)

            primary_zone = data["zones"][0] if data["zones"] else "unknown"

            avg_conf = data.get("conf_sum", 0.7) / max(data.get("frame_count", 1), 1)

            # Calculate average behavior metrics
            behavior_scores = data.get("behavior_scores", [])
            behavior_types = data.get("behavior_types", [])
            body_orientations = data.get("body_orientations", [])
            postures = data.get("postures", [])

            avg_engagement = round(sum(behavior_scores) / len(behavior_scores), 1) if behavior_scores else None
            avg_orientation = round(sum(body_orientations) / len(body_orientations), 2) if body_orientations else None
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
                "engagement_score": avg_engagement,
                "behavior_type": dominant_behavior,
                "body_orientation": avg_orientation,
                "posture": dominant_posture,
            })

            # Create additional events for zone transitions
            # Distribute time evenly across zones
            zones_after_first = data["zones"][1:]
            if zones_after_first and dwell_seconds > 0:
                zone_interval = dwell_seconds / (len(zones_after_first) + 1)
                for idx, zone in enumerate(zones_after_first, 1):
                    transition_time = data["first_seen"] + timedelta(seconds=zone_interval * idx)
                    zone_dwell = zone_interval  # Each zone gets equal share
                    all_events.append({
                        "venue_id": venue_id,
                        "pseudo_id": visitor_id,
                        "timestamp": transition_time.isoformat(),
                        "zone": zone,
                        "dwell_seconds": round(zone_dwell, 1),
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
                        embedding_bytes=_reid.serialize_embedding(visitor["embedding"]),
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
        print(f"Pipeline error for job {job_id}: {traceback.format_exc()}")
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
