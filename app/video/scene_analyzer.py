"""
Scene Analyzer
==============
Pre-analyzes video frames to classify scene conditions (brightness, camera motion,
crowd density) and dynamically generates tracker/detection parameters.
"""

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import yaml


class PhoneFootageError(Exception):
    """Raised when phone/panning footage is detected."""
    def __init__(self, message, code="PHONE_FOOTAGE"):
        self.code = code
        super().__init__(message)


@dataclass
class SceneProfile:
    avg_brightness: float       # 0-255 mean pixel intensity
    brightness_std: float       # variation across sampled frames
    motion_score: float         # 0+, estimated camera motion magnitude (px)
    crowd_density: float        # 0-1, fraction of frame area covered by person bboxes
    max_detections: int         # peak raw detection count across sampled frames
    peak_frame_idx: int         # which sampled frame had max detections
    is_dark: bool               # brightness < 120
    is_shaky: bool              # motion_score > 3.0
    is_crowded: bool            # max_detections >= 8 or crowd_density > 0.3
    fps: float
    duration_seconds: float
    aspect_ratio: float = 0.0          # width/height (phone: ~0.56 or ~1.78)
    global_motion_ratio: float = 0.0   # global vs local motion (>0.7 = panning)
    is_phone_footage: bool = False


def analyze_scene(cap, model, total_frames, fps, frame_width, frame_height):
    """Sample frames across the video and classify scene conditions.

    Uses sequential read with frame skipping (avoids slow I-frame seeking).
    Downsamples to 320x240 for optical flow (performance).
    Resets VideoCapture to frame 0 after analysis.
    """
    from app.video.deps import cv2 as _cv2

    duration = total_frames / max(fps, 1)

    # Target sample positions: 0%, 15%, 30%, 50%, 70%, 85%, 95%
    sample_pcts = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
    sample_indices = [min(int(p * total_frames), total_frames - 1) for p in sample_pcts]

    # Sequential read through video, grabbing only at sample positions
    cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
    sampled_frames = []
    sampled_gray_small = []  # Downsampled for optical flow
    brightnesses = []
    detection_counts = []
    density_ratios = []
    peak_detections = 0
    peak_idx = 0

    frame_idx = 0
    next_sample = 0

    while next_sample < len(sample_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == sample_indices[next_sample]:
            sampled_frames.append(frame)

            # Brightness
            gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
            brightnesses.append(float(np.mean(gray)))

            # Downsampled grayscale for optical flow
            small = _cv2.resize(gray, (320, 240))
            sampled_gray_small.append(small)

            # Crowd density: raw detection (no tracking)
            results = model(frame, classes=[0], conf=0.20, verbose=False)
            boxes = results[0].boxes
            n_det = len(boxes) if boxes is not None else 0
            detection_counts.append(n_det)

            if n_det > peak_detections:
                peak_detections = n_det
                peak_idx = frame_idx

            # Compute bbox area ratio
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                total_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in xyxy)
                frame_area = frame_width * frame_height
                density_ratios.append(total_area / frame_area)
            else:
                density_ratios.append(0.0)

            next_sample += 1
        frame_idx += 1

    # Compute motion score via optical flow between consecutive samples
    # Also separate global (camera) motion from local (person) motion
    motion_magnitudes = []
    global_motion_ratios = []
    for i in range(1, len(sampled_gray_small)):
        prev = sampled_gray_small[i - 1]
        curr = sampled_gray_small[i]
        flow = _cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = _cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        motion_magnitudes.append(mean_mag)

        # Global motion: median flow (camera pan moves everything uniformly)
        # Local motion: std of flow (people moving independently varies spatially)
        median_flow_x = float(np.median(flow[..., 0]))
        median_flow_y = float(np.median(flow[..., 1]))
        global_mag = (median_flow_x**2 + median_flow_y**2)**0.5
        local_mag = float(np.std(mag))
        if mean_mag > 0.5:  # Only compute ratio when there's meaningful motion
            global_motion_ratios.append(global_mag / (global_mag + local_mag + 1e-6))

    avg_brightness = float(np.mean(brightnesses)) if brightnesses else 128.0
    brightness_std = float(np.std(brightnesses)) if len(brightnesses) > 1 else 0.0
    motion_score = float(np.mean(motion_magnitudes)) if motion_magnitudes else 0.0
    max_density = max(density_ratios) if density_ratios else 0.0
    aspect_ratio = frame_width / max(frame_height, 1)
    global_motion_ratio = float(np.mean(global_motion_ratios)) if global_motion_ratios else 0.0

    # Detect phone/panning footage
    is_phone = _detect_phone_footage(motion_score, aspect_ratio, global_motion_ratio)

    # Reset video to start
    cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)

    return SceneProfile(
        avg_brightness=avg_brightness,
        brightness_std=brightness_std,
        motion_score=motion_score,
        crowd_density=max_density,
        max_detections=peak_detections,
        peak_frame_idx=peak_idx,
        is_dark=avg_brightness < 120,
        is_shaky=motion_score > 3.0,
        is_crowded=peak_detections >= 8 or max_density > 0.3,
        fps=fps,
        duration_seconds=duration,
        aspect_ratio=round(aspect_ratio, 3),
        global_motion_ratio=round(global_motion_ratio, 3),
        is_phone_footage=is_phone,
    )


# Phone/panning detection thresholds
_PHONE_ASPECT_RATIOS = {
    (9, 16), (16, 9),   # Standard phone vertical/horizontal
    (3, 4), (4, 3),     # Older phones / tablets
}
_PHONE_SHAKY_THRESHOLD = 5.0       # Higher than normal 3.0
_PHONE_PANNING_THRESHOLD = 0.55    # Global vs local motion ratio (>0.55 = camera dominates)
_PHONE_ASPECT_TOLERANCE = 0.08     # How close to exact phone ratio


def _is_phone_aspect_ratio(aspect_ratio):
    """Check if aspect ratio matches common phone ratios."""
    for w, h in _PHONE_ASPECT_RATIOS:
        target = w / h
        if abs(aspect_ratio - target) < _PHONE_ASPECT_TOLERANCE:
            return True
    return False


def _detect_phone_footage(motion_score, aspect_ratio, global_motion_ratio):
    """Detect phone/panning footage based on scene analysis metrics.

    Returns True if footage appears to be handheld phone with significant panning.
    """
    is_phone_ratio = _is_phone_aspect_ratio(aspect_ratio)
    is_very_shaky = motion_score > _PHONE_SHAKY_THRESHOLD
    is_panning = global_motion_ratio > _PHONE_PANNING_THRESHOLD

    # Must have phone aspect ratio AND (very shaky OR dominant panning motion)
    return is_phone_ratio and (is_very_shaky or is_panning)


def check_phone_footage(profile):
    """Raise PhoneFootageError if footage is detected as phone/panning video.

    Call this after analyze_scene() but before starting the main processing loop.
    """
    if profile.is_phone_footage:
        raise PhoneFootageError(
            f"Phone/panning footage detected (aspect_ratio={profile.aspect_ratio:.2f}, "
            f"motion={profile.motion_score:.1f}, panning_ratio={profile.global_motion_ratio:.2f}). "
            f"Use static CCTV camera for accurate visitor counting."
        )


def generate_tracker_config(profile):
    """Generate BoT-SORT config tuned to the analyzed scene.

    Priority-ordered combinatorial matching for scene conditions.
    Returns a dict suitable for writing to YAML.
    """
    base = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "track_buffer": 150,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": False,
        "model": "auto",
    }

    dark = profile.is_dark
    shaky = profile.is_shaky
    crowded = profile.is_crowded

    if dark and shaky and crowded:
        base["new_track_thresh"] = 0.3
        base["track_high_thresh"] = 0.3
        base["track_low_thresh"] = 0.08
        base["track_buffer"] = 200
        base["match_thresh"] = 0.85
        base["proximity_thresh"] = 0.4
    elif dark and shaky:
        base["new_track_thresh"] = 0.3
        base["track_high_thresh"] = 0.3
        base["track_low_thresh"] = 0.08
        base["track_buffer"] = 200
        base["match_thresh"] = 0.85
    elif dark and crowded:
        base["new_track_thresh"] = 0.3
        base["track_high_thresh"] = 0.3
        base["track_low_thresh"] = 0.08
        base["proximity_thresh"] = 0.4
    elif dark:
        base["new_track_thresh"] = 0.3
        base["track_high_thresh"] = 0.3
        base["track_low_thresh"] = 0.08
    elif shaky and crowded:
        base["new_track_thresh"] = 0.5
        base["track_high_thresh"] = 0.4
        base["track_buffer"] = 200
        base["proximity_thresh"] = 0.4
    elif shaky:
        base["new_track_thresh"] = 0.5
        base["track_high_thresh"] = 0.5
        base["track_buffer"] = 200
    elif crowded:
        base["new_track_thresh"] = 0.5
        base["track_high_thresh"] = 0.4
        base["proximity_thresh"] = 0.4
    # else: bright, static, sparse — use defaults

    return base


def compute_frame_interval(profile):
    """Compute adaptive frame sampling interval based on scene conditions."""
    base = max(1, int(profile.fps / 2))  # Default: every 0.5s

    if profile.is_crowded and profile.is_shaky:
        return max(1, int(profile.fps / 3))      # Every 0.33s
    elif profile.is_crowded:
        return max(1, int(profile.fps / 2.5))    # Every 0.4s
    elif profile.is_shaky:
        return max(1, int(profile.fps / 3))      # Every 0.33s

    return base


def compute_track_score(frame_count, avg_conf):
    """Score a track by combining length and confidence."""
    return frame_count * avg_conf


def is_valid_track(frame_count, avg_conf, profile):
    """Determine if a track represents a real person using confidence-weighted scoring.

    High-confidence short tracks (2 frames, conf 0.8 -> score 1.6) count.
    Low-confidence long tracks (10 frames, conf 0.26 -> score 2.6) count.
    Only filter genuinely noisy detections.
    """
    # Always reject single-frame tracks with very low confidence
    if frame_count == 1 and avg_conf < 0.3:
        return False

    score = compute_track_score(frame_count, avg_conf)

    if profile.is_dark or profile.is_crowded:
        return score >= 0.5
    return score >= 1.0


def peak_density_floor(profile, tracked_count):
    """Ensure visitor count is at least as high as peak simultaneous detections.

    If 10 people are visible in a single frame, there must be at least 10 unique visitors.
    """
    return max(tracked_count, profile.max_detections)


def write_temp_tracker_yaml(config):
    """Write tracker config to a temporary YAML file for Ultralytics.

    Returns the file path. Caller must clean up in a finally block.
    """
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="botsort_adaptive_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return path
