"""
extract_frames.py
Extracts frames from all videos in a directory using OpenCV.

1. Builds manifest.json mapping short hash IDs → absolute video paths.
2. Extracts frames named {hash}-{MM:SS}.jpg every N seconds.
3. Skips frames that are too similar to the last saved frame using
   histogram comparison (avoids storing near-identical static shots).
4. Processes multiple videos in parallel using multiprocessing.

Can be run standalone:
  python extract_frames.py ./my_videos --output ./output --interval 3
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: opencv-python and numpy are required.")
    print("  pip install opencv-python-headless numpy")
    raise

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".flv")
DEFAULT_INTERVAL = 3
DEFAULT_HEAD_SKIP = 5
DEFAULT_TAIL_SKIP = 10
DEFAULT_SIM_THRESHOLD = 0.98  # skip frame if histogram similarity > this
HASH_LENGTH = 8
HIST_RESIZE = (320, 180)  # downscale before histogram comparison for speed


def _hash_path(abs_path: str) -> str:
    """Short deterministic hash of an absolute file path."""
    return hashlib.sha256(abs_path.encode()).hexdigest()[:HASH_LENGTH]


def _timestamp_str(seconds: float) -> str:
    """Return MM:SS string with zero-padding (e.g. 3 -> '00:03')."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _hist_similarity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Compare two frames using HSV histogram correlation.
    Downscales first for speed. Returns a value in [-1, 1] where 1 = identical.
    """
    small_a = cv2.resize(frame_a, HIST_RESIZE)
    small_b = cv2.resize(frame_b, HIST_RESIZE)

    hsv_a = cv2.cvtColor(small_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(small_b, cv2.COLOR_BGR2HSV)

    # compute histogram on H and S channels (ignore V to be lighting-robust)
    h_bins, s_bins = 50, 60
    channels = [0, 1]
    ranges = [0, 180, 0, 256]

    hist_a = cv2.calcHist([hsv_a], channels, None, [h_bins, s_bins], ranges)
    hist_b = cv2.calcHist([hsv_b], channels, None, [h_bins, s_bins], ranges)

    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)


def _build_manifest(videos: list[Path], output_dir: Path) -> dict[str, str]:
    """
    Build or load manifest.json mapping hash → absolute video path.
    Merges new videos in (idempotent).
    """
    manifest_path = output_dir / "manifest.json"

    manifest: dict[str, str] = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    existing_paths = set(manifest.values())

    for video in videos:
        abs_path = str(video.resolve())
        if abs_path in existing_paths:
            continue
        vid_hash = _hash_path(abs_path)
        while vid_hash in manifest and manifest[vid_hash] != abs_path:
            vid_hash = hashlib.sha256(
                (abs_path + vid_hash).encode()
            ).hexdigest()[:HASH_LENGTH]
        manifest[vid_hash] = abs_path

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest: {len(manifest)} video(s) → {manifest_path}")
    return manifest


def _extract_video_frames(args: tuple) -> tuple[str, int, int]:
    """
    Extract frames from a single video. Designed to be called via multiprocessing.
    Takes a single tuple arg for Pool.map compatibility.
    Returns (video_name, saved_count, skipped_count).
    """
    video_path, vid_hash, frames_dir, interval, head_skip, tail_skip, sim_threshold, force = args
    video_path = Path(video_path)
    frames_dir = Path(frames_dir)

    existing = [f for f in os.listdir(frames_dir) if f.startswith(vid_hash + "-")]
    if existing and not force:
        print(f"  [{vid_hash}] Skip {video_path.name} ({len(existing)} frames exist)")
        return video_path.name, len(existing), 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [{vid_hash}] [WARN] Cannot open {video_path.name}")
        return video_path.name, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = n_frames / fps if fps else 0

    if duration <= head_skip + tail_skip:
        print(f"  [{vid_hash}] Skip {video_path.name}: {duration:.1f}s ≤ {head_skip}s head + {tail_skip}s tail skip")
        cap.release()
        return video_path.name, 0, 0

    print(f"  [{vid_hash}] {video_path.name} ({duration:.0f}s)")

    saved = 0
    skipped = 0
    last_saved_frame = None
    t = head_skip

    while t <= duration - tail_skip:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        # histogram similarity check against last saved frame
        if last_saved_frame is not None:
            sim = _hist_similarity(last_saved_frame, frame)
            if sim > sim_threshold:
                skipped += 1
                t += interval
                continue

        ts = _timestamp_str(t)
        out_path = frames_dir / f"{vid_hash}-{ts}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        last_saved_frame = frame
        saved += 1
        t += interval

    cap.release()
    print(f"  [{vid_hash}] → {saved} saved, {skipped} skipped")
    return video_path.name, saved, skipped


def extract_frames(
    video_dir: str,
    output_dir: str = "output",
    interval: float = DEFAULT_INTERVAL,
    head_skip: float = DEFAULT_HEAD_SKIP,
    tail_skip: float = DEFAULT_TAIL_SKIP,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    force: bool = False,
    workers: int = 0,
) -> str | None:
    """
    Extract frames from all videos in video_dir.

    1. Builds manifest.json (hash → absolute video path)
    2. Extracts frames as {hash}-{MM:SS}.jpg, skipping near-duplicates
    3. Processes videos in parallel using multiprocessing

    Returns the frames directory path, or None if no videos found.
    """
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        print(f"Error: video directory not found: {video_dir}")
        return None

    videos = sorted(
        f for f in video_dir.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"No video files found in {video_dir}")
        return None

    output_path = Path(output_dir)
    frames_dir = output_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: build manifest
    manifest = _build_manifest(videos, output_path)
    path_to_hash = {v: k for k, v in manifest.items()}

    # Determine worker count
    if workers <= 0:
        workers = min(len(videos), cpu_count())

    print(f"Found {len(videos)} video(s) in {video_dir}")
    print(f"Workers: {workers} | Interval: {interval}s | Head skip: {head_skip}s | Tail skip: {tail_skip}s | Sim: {sim_threshold}")
    print(f"Output: {frames_dir}/")

    # Build job list
    jobs = []
    for vf in videos:
        abs_path = str(vf.resolve())
        vid_hash = path_to_hash.get(abs_path)
        if not vid_hash:
            print(f"  [WARN] {vf.name} not in manifest, skipping")
            continue
        jobs.append((
            str(vf), vid_hash, str(frames_dir),
            interval, head_skip, tail_skip, sim_threshold, force,
        ))

    # Process in parallel
    total_saved = 0
    total_skipped = 0

    if workers == 1:
        for job in jobs:
            name, saved, skipped = _extract_video_frames(job)
            total_saved += saved
            total_skipped += skipped
    else:
        with Pool(processes=workers) as pool:
            results = pool.map(_extract_video_frames, jobs)
        for name, saved, skipped in results:
            total_saved += saved
            total_skipped += skipped

    print(f"\nTotal: {total_saved} frames saved, {total_skipped} skipped → {frames_dir}/")
    return str(frames_dir) if total_saved > 0 else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("video_dir", help="Directory containing video files")
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"Seconds between frames (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--head-skip", type=float, default=DEFAULT_HEAD_SKIP,
                        help=f"Skip first N seconds of each video (default: {DEFAULT_HEAD_SKIP})")
    parser.add_argument("--tail-skip", type=float, default=DEFAULT_TAIL_SKIP,
                        help=f"Skip last N seconds of each video (default: {DEFAULT_TAIL_SKIP})")
    parser.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD,
                        help=f"Skip frame if histogram similarity > this (default: {DEFAULT_SIM_THRESHOLD})")
    parser.add_argument("-j", "--workers", type=int, default=0,
                        help="Number of parallel workers (default: auto, uses all CPU cores)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    extract_frames(
        args.video_dir, args.output, args.interval,
        args.head_skip, args.tail_skip, args.sim_threshold,
        args.force, args.workers,
    )