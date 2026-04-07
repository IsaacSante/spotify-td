"""
run.py — master script for lyric-to-image pipeline

Runs the full pipeline end-to-end:
  1. Extract frames from videos in a folder
  2. Generate CLIP embeddings for all frames

Usage:
  python run.py ./my_videos
  python run.py ./my_videos --interval 1 --batch-size 32
  python run.py ./my_videos --output ./my_output

The video folder can contain any mix of .mp4, .mkv, .webm, .avi, .mov files
from any source (YouTube downloads, screen recordings, stock footage, etc.)
"""

import argparse
import sys
from extract_frames import extract_frames
from generate_embeddings import generate_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="lyric-to-image: video frames → CLIP embeddings"
    )
    parser.add_argument(
        "video_dir",
        help="Path to folder containing video files",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for frames/ and embeddings/ (default: ./output)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Seconds between extracted frames (default: 3)",
    )
    parser.add_argument(
        "--tail-skip",
        type=float,
        default=10.0,
        help="Skip last N seconds of each video (default: 10)",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.98,
        help="Skip frame if histogram similarity to last saved frame > this (default: 0.98)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for CLIP embedding generation (default: 16)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract frames and regenerate embeddings even if they exist",
    )
    args = parser.parse_args()

    # Step 1: extract frames
    print("=" * 60)
    print("STEP 1: Extracting frames")
    print("=" * 60)
    frames_dir = extract_frames(
        video_dir=args.video_dir,
        output_dir=args.output,
        interval=args.interval,
        tail_skip=args.tail_skip,
        sim_threshold=args.sim_threshold,
        force=args.force,
    )

    if not frames_dir:
        print("No frames extracted. Exiting.")
        sys.exit(1)

    # Step 2: generate embeddings
    print()
    print("=" * 60)
    print("STEP 2: Generating CLIP embeddings")
    print("=" * 60)
    generate_embeddings(
        frames_dir=frames_dir,
        output_dir=args.output,
        batch_size=args.batch_size,
        force=args.force,
    )

    print()
    print("=" * 60)
    print("DONE")
    print(f"  Frames:     {frames_dir}/")
    print(f"  Embeddings: {args.output}/embeddings/image_embeddings.npy")
    print(f"  Paths:      {args.output}/embeddings/image_paths.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()