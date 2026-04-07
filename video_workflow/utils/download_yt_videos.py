#!/usr/bin/env python3
"""
utils/download_yt_videos.py
Download videos from YouTube URLs or playlists using yt-dlp.

Downloads video-only (no audio) at up to 1080p, encoded as H.264 mp4
for compatibility with QuickTime and TouchDesigner.

Usage:
  python utils/download_yt_videos.py "https://youtube.com/watch?v=xxxxx" ./my_videos
  python utils/download_yt_videos.py "https://youtube.com/playlist?list=PLxxxxx" ./my_videos
  python utils/download_yt_videos.py urls.txt ./my_videos

Accepts:
  - A single YouTube URL (video or playlist)
  - A .txt file with one URL per line

Requires:
  brew install yt-dlp ffmpeg
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def download(url: str, output_dir: str):
    """Download a single URL (video or playlist) to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][ext=mp4][vcodec^=avc1]/"
                    "bestvideo[height<=1080][ext=mp4]/"
                    "bestvideo[height<=1080]",
        "--no-audio",
        "--merge-output-format", "mp4",
        "--postprocessor-args", "ffmpeg:-an -c:v libx264 -preset fast -crf 18",
        "--output", os.path.join(output_dir, "%(title)s.%(ext)s"),
        "--restrict-filenames",
        "--no-overwrites",
        url,
    ]

    print(f"Downloading: {url}")
    print(f"Output dir:  {output_dir}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos via yt-dlp")
    parser.add_argument(
        "source",
        help="YouTube URL (video or playlist) or path to a .txt file with one URL per line",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save downloaded videos",
    )
    args = parser.parse_args()

    # check yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: yt-dlp not found. Install with: brew install yt-dlp")
        sys.exit(1)

    # if source is a .txt file, read URLs from it
    source_path = Path(args.source)
    if source_path.exists() and source_path.suffix == ".txt":
        urls = [
            line.strip()
            for line in source_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        print(f"Found {len(urls)} URL(s) in {source_path}")
    else:
        urls = [args.source]

    for url in urls:
        try:
            download(url, args.output_dir)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Failed to download {url}: {e}")
            continue

    print("Done.")


if __name__ == "__main__":
    main()