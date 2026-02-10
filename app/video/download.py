"""
Video Download
==============
YouTube and URL video downloading.
"""

from pathlib import Path

from app.video.deps import load_video_deps, yt_dlp


def download_youtube_video(url: str, output_path: str) -> str:
    """Download YouTube video to file."""
    load_video_deps()
    from app.video.deps import yt_dlp as _yt_dlp

    ydl_opts = {
        'format': 'best[height<=720]/best[height<=1080]/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4',
    }

    with _yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the actual downloaded file
    base = Path(output_path).stem
    parent = Path(output_path).parent
    for f in parent.iterdir():
        if f.stem.startswith(base):
            return str(f)

    return output_path
