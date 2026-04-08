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


def load(output_dir: str, model_name: str = DEFAULT_MODEL):
    global model, processor, image_embeddings, image_paths, manifest

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


def _encode_text(text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
    emb = features.numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb


@app.route("/lookup", methods=["GET"])
def lookup():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "missing ?q= parameter"}), 400

    top_k = request.args.get("top", 1, type=int)
    top_k = max(1, min(top_k, 50))

    text_emb = _encode_text(query)
    scores = (text_emb @ image_embeddings.T)[0]
    top_idx = np.argsort(scores)[-top_k:][::-1]

    results = []
    for i in top_idx:
        result = _parse_frame_path(image_paths[i])
        result["score"] = float(scores[i])
        results.append(result)

    if top_k == 1:
        return jsonify({"query": query, **results[0]})
    return jsonify({"query": query, "results": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "frames": len(image_paths) if image_paths else 0,
        "videos": len(manifest) if manifest else 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lyric-to-image lookup server")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory from the build pipeline (default: ./output)")
    parser.add_argument("--port", type=int, default=8976)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    load(args.output, args.model)
    app.run(host=args.host, port=args.port, debug=False)