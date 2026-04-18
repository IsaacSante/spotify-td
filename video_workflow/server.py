"""
server.py
Lightweight lookup server. Loads SigLIP 2 + embeddings once on startup,
takes a word query, returns the source video path and timestamp.

Usage:
  python server.py                          # defaults: output dir = ./output, port 8976
  python server.py --output ./my_output
  python server.py --port 9000

Query:
  GET http://localhost:8976/lookup?q=fire
  → {"query": "fire", "video": "/abs/path/to/nature.mp4", "timestamp": "01:24"}

  GET http://localhost:8976/lookup?q=ocean&top=3
  → {"query": "ocean", "results": [{"video": "...", "timestamp": "00:06"}, ...]}

  GET http://localhost:8976/lookup_converted?q=fire
  → {"query": "fire", "video": "/abs/path/to/videos_hap/nature.mov", "timestamp": "01:24"}
"""

import argparse
import json
import os
import re
import numpy as np
import pickle
import torch
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoProcessor

app = Flask(__name__)

DEFAULT_MODEL = "google/siglip2-so400m-patch14-384"

# globals loaded on startup
model = None
processor = None
image_embeddings = None
image_paths = None
manifest = None  # hash → absolute video path
HAP_DIR = None   # directory with HAP-converted .mov files


def load(output_dir: str, model_name: str = DEFAULT_MODEL, hap_dir: str = "videos_hap"):
    global model, processor, image_embeddings, image_paths, manifest, HAP_DIR

    emb_file = os.path.join(output_dir, "embeddings", "image_embeddings.npy")
    paths_file = os.path.join(output_dir, "embeddings", "image_paths.pkl")
    manifest_file = os.path.join(output_dir, "manifest.json")

    for f, label in [(emb_file, "Embeddings"), (paths_file, "Paths"), (manifest_file, "Manifest")]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"{label} not found: {f}. Run the pipeline first.")

    print(f"Loading model: {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    print("Loading embeddings...")
    image_embeddings = np.load(emb_file)
    with open(paths_file, "rb") as f:
        image_paths = pickle.load(f)

    print("Loading manifest...")
    with open(manifest_file) as f:
        manifest = json.load(f)

    # Resolve HAP directory
    if os.path.isabs(hap_dir):
        resolved_hap = hap_dir
    else:
        resolved_hap = os.path.join(os.path.dirname(os.path.abspath(output_dir)), hap_dir)

    if os.path.isdir(resolved_hap):
        HAP_DIR = os.path.abspath(resolved_hap)
        hap_count = len([f for f in os.listdir(HAP_DIR) if f.endswith(".mov")])
        print(f"HAP directory: {HAP_DIR} ({hap_count} files)")
    else:
        print(f"HAP directory not found: {resolved_hap} (/lookup_converted will be unavailable)")

    print(f"Ready. {len(image_paths)} frames from {len(manifest)} video(s).")


def _parse_frame_path(frame_path: str) -> dict:
    """
    Parse a frame filename like 'a3f2b1c4-01:24.jpg' into
    {"video": "/abs/path/to/video.mp4", "timestamp": "01:24"}
    """
    basename = os.path.splitext(os.path.basename(frame_path))[0]
    match = re.match(r"^([a-f0-9]+)-(\d{2}:\d{2})$", basename)
    if not match:
        return {"video": None, "timestamp": None}

    vid_hash = match.group(1)
    timestamp = match.group(2)
    video_path = manifest.get(vid_hash)

    return {"video": video_path, "timestamp": timestamp, "frame": os.path.basename(frame_path)}


def _swap_to_hap(result: dict) -> dict:
    """Replace the video path with the HAP .mov equivalent if it exists."""
    if result["video"] and HAP_DIR:
        stem = os.path.splitext(os.path.basename(result["video"]))[0]
        hap_path = os.path.join(HAP_DIR, stem + ".mov")
        if os.path.exists(hap_path):
            result["video"] = hap_path
    return result


def _encode_text(text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
    emb = features.numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb


def _do_lookup(query: str, top_k: int) -> list[dict]:
    """Shared lookup logic for both routes."""
    text_emb = _encode_text(query)
    scores = (text_emb @ image_embeddings.T)[0]
    top_idx = np.argsort(scores)[-top_k:][::-1]

    results = []
    for i in top_idx:
        result = _parse_frame_path(image_paths[i])
        result["score"] = float(scores[i])
        results.append(result)
    return results


@app.route("/lookup", methods=["GET"])
def lookup():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "missing ?q= parameter"}), 400

    top_k = request.args.get("top", 1, type=int)
    top_k = max(1, min(top_k, 50))

    results = _do_lookup(query, top_k)

    if top_k == 1:
        return jsonify({"query": query, **results[0]})
    return jsonify({"query": query, "results": results})


@app.route("/lookup_converted", methods=["GET"])
def lookup_converted():
    if not HAP_DIR:
        return jsonify({"error": "HAP directory not configured"}), 503

    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "missing ?q= parameter"}), 400

    top_k = request.args.get("top", 1, type=int)
    top_k = max(1, min(top_k, 50))

    results = [_swap_to_hap(r) for r in _do_lookup(query, top_k)]

    if top_k == 1:
        return jsonify({"query": query, **results[0]})
    return jsonify({"query": query, "results": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "frames": len(image_paths) if image_paths else 0,
        "videos": len(manifest) if manifest else 0,
        "hap_dir": HAP_DIR,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lyric-to-image lookup server")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory from the build pipeline (default: ./output)")
    parser.add_argument("--port", type=int, default=8976)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--hap-dir", default="videos_hap",
                        help="Directory with HAP-converted .mov files (default: ./videos_hap)")
    args = parser.parse_args()

    load(args.output, args.model, args.hap_dir)
    app.run(host=args.host, port=args.port, debug=False)