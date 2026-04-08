"""
run.py — master script for lyric-to-image pipeline

Runs the full pipeline end-to-end:
  1. Extract frames from videos in a folder (parallel via multiprocessing)
  2. Generate SigLIP 2 embeddings for all frames

Usage:
  python run.py ./my_videos
  python run.py ./my_videos --interval 1 --batch-size 32
  python run.py ./my_videos --output ./my_output

The video folder can contain any mix of .mp4, .mkv, .webm, .avi, .mov files
from any source (YouTube downloads, screen recordings, stock footage, etc.)
"""

import argparse
import sys
from extract_frames import extract_frames, DEFAULT_INTERVAL, DEFAULT_HEAD_SKIP, DEFAULT_TAIL_SKIP, DEFAULT_SIM_THRESHOLD
from generate_embeddings import generate_embeddings, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(
        description="lyric-to-image: video frames → SigLIP 2 embeddings"
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
        default=DEFAULT_INTERVAL,
        help=f"Seconds between extracted frames (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--head-skip",
        type=float,
        default=DEFAULT_HEAD_SKIP,
        help=f"Skip first N seconds of each video (default: {DEFAULT_HEAD_SKIP})",
    )
    parser.add_argument(
        "--tail-skip",
        type=float,
        default=DEFAULT_TAIL_SKIP,
        help=f"Skip last N seconds of each video (default: {DEFAULT_TAIL_SKIP})",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=DEFAULT_SIM_THRESHOLD,
        help=f"Skip frame if histogram similarity to last saved frame > this (default: {DEFAULT_SIM_THRESHOLD})",
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=0,
        help="Number of parallel workers for frame extraction (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for SigLIP embedding generation (default: 32)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model for embeddings (default: {DEFAULT_MODEL})",
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
        head_skip=args.head_skip,
        tail_skip=args.tail_skip,
        sim_threshold=args.sim_threshold,
        force=args.force,
        workers=args.workers,
    )

    if not frames_dir:
        print("No frames extracted. Exiting.")
        sys.exit(1)

    # Step 2: generate embeddings
    print()
    print("=" * 60)
    print("STEP 2: Generating SigLIP 2 embeddings")
    print("=" * 60)
    generate_embeddings(
        frames_dir=frames_dir,
        output_dir=args.output,
        batch_size=args.batch_size,
        model_name=args.model,
        force=args.force,
    )

    print()
    print("=" * 60)
    print("DONE")
    print(f"  Frames:     {frames_dir}/")
    print(f"  Embeddings: {args.output}/embeddings/image_embeddings.npy")
    print(f"  Paths:      {args.output}/embeddings/image_paths.pkl")
    print()
    print("Start the lookup server:")
    print(f"  python server.py --output {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()