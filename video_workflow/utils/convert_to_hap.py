#!/usr/bin/env python3
"""
convert_to_hap.py
Converts all videos referenced in manifest.json to HAP codec (.mov)
for GPU-accelerated real-time scrubbing in TouchDesigner.

Lives in: utils/convert_to_hap.py
Outputs to: videos_hap/ (sibling to videos/)

Usage (from video_workflow root):
  python utils/convert_to_hap.py
  python utils/convert_to_hap.py --hap-type hap_q
  python utils/convert_to_hap.py --manifest output/manifest.json --output-dir videos_hap
"""

import json
import subprocess
import argparse
import shutil
from pathlib import Path


def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Install with: brew install ffmpeg")
        raise SystemExit(1)


def convert_all(
    manifest_path: str = "output/manifest.json",
    output_dir: str = "videos_hap",
    hap_type: str = "hap_q",
):
    check_ffmpeg()

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}")
        raise SystemExit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = len(manifest)
    converted = 0
    skipped = 0
    errors = 0

    print(f"Converting {total} video(s) to HAP ({hap_type}) → {out}/\n")

    for vid_hash, src_path in manifest.items():
        src = Path(src_path)
        dst = out / (src.stem + ".mov")

        if not src.exists():
            print(f"  [{vid_hash}] MISSING: {src.name}")
            errors += 1
            continue

        if dst.exists():
            print(f"  [{vid_hash}] Skip (exists): {dst.name}")
            skipped += 1
            continue

        print(f"  [{vid_hash}] {src.name} → {dst.name}")
        cmd = [
            "ffmpeg",
            "-i", str(src),
            "-vcodec", "hap",
            "-format", hap_type,    # hap_q, hap_alpha, or hap
            "-an",
            "-y",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"           ERROR: {result.stderr.strip()[-200:]}")
            errors += 1
        else:
            converted += 1

    print(f"\nDone: {converted} converted, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert manifest videos to HAP codec for TouchDesigner"
    )
    parser.add_argument(
        "--manifest", default="output/manifest.json",
        help="Path to manifest.json (default: output/manifest.json)",
    )
    parser.add_argument(
        "--output-dir", default="videos_hap",
        help="Output directory for .mov files (default: videos_hap/)",
    )
    parser.add_argument(
        "--hap-type", default="hap_q",
        choices=["hap", "hap_alpha", "hap_q"],
        help="HAP variant: hap (fast, larger), hap_q (snappy+quality), hap_alpha (with alpha channel)",
    )
    args = parser.parse_args()
    convert_all(args.manifest, args.output_dir, args.hap_type)