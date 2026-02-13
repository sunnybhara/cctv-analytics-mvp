"""Step 1: Extract training frames from bar footage.

Usage:
    python extract_frames.py                          # from RAW_VIDEO_DIR
    python extract_frames.py --source /path/to/videos
    python extract_frames.py --include-confirmed      # also pull from confirmed_clips/
"""
import os
import glob
import subprocess
import argparse
from config import FRAMES_DIR, RAW_VIDEO_DIR, CONFIRMED_CLIPS_DIR, EXTRACT_FPS, MAX_FRAMES


def extract_from_video(video_path: str, output_dir: str, fps: float, prefix: str) -> int:
    """Extract frames from a single video using ffmpeg.
    Returns number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, f"{prefix}_%05d.jpg")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high quality JPEG
        pattern,
        "-y", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)

    extracted = glob.glob(os.path.join(output_dir, f"{prefix}_*.jpg"))
    return len(extracted)


def main():
    parser = argparse.ArgumentParser(description="Extract frames from bar footage")
    parser.add_argument("--source", default=RAW_VIDEO_DIR, help="Video directory")
    parser.add_argument("--output", default=FRAMES_DIR, help="Output frames directory")
    parser.add_argument("--fps", type=float, default=EXTRACT_FPS, help="Frames per second")
    parser.add_argument("--include-confirmed", action="store_true",
                        help="Also extract from confirmed_clips/")
    args = parser.parse_args()

    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    videos = []
    for ext in video_extensions:
        videos.extend(glob.glob(os.path.join(args.source, ext)))

    if args.include_confirmed and os.path.exists(CONFIRMED_CLIPS_DIR):
        for ext in video_extensions:
            videos.extend(glob.glob(os.path.join(CONFIRMED_CLIPS_DIR, ext)))

    if not videos:
        print(f"No videos found in {args.source}")
        return

    total_frames = 0
    for i, video in enumerate(sorted(videos)):
        name = os.path.splitext(os.path.basename(video))[0]
        prefix = f"vid{i:03d}_{name}"
        count = extract_from_video(video, args.output, args.fps, prefix)
        total_frames += count
        print(f"[{i+1}/{len(videos)}] {video}: {count} frames")

        if total_frames >= MAX_FRAMES:
            print(f"Reached MAX_FRAMES={MAX_FRAMES}, stopping")
            break

    print(f"\nTotal: {total_frames} frames in {args.output}")


if __name__ == "__main__":
    main()
