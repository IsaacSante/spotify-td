"""
generate_embeddings.py
Generates SigLIP 2 embeddings for all images in a frames directory.

Uses google/siglip2-so400m-patch14-384 for rich semantic understanding
of visual content (composition, mood, style, action, spatial layout).

Outputs:
  {output_dir}/embeddings/image_embeddings.npy  — (N, 1152) float32, L2-normalized
  {output_dir}/embeddings/image_paths.pkl        — list of N file paths

Can be run standalone:
  python generate_embeddings.py ./output/frames --output ./output --batch-size 16
"""

import os
import numpy as np
import pickle
import argparse
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_MODEL = "google/siglip2-so400m-patch14-384"


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_images(root_dir: str) -> list[str]:
    """Find all images in a directory (flat — no recursion needed)."""
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def generate_embeddings(
    frames_dir: str,
    output_dir: str = "output",
    batch_size: int = 16,
    model_name: str = DEFAULT_MODEL,
    force: bool = False,
):
    """
    Generate SigLIP 2 embeddings for all images in frames_dir.

    Saves image_embeddings.npy and image_paths.pkl to {output_dir}/embeddings/.
    """
    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    emb_file = os.path.join(emb_dir, "image_embeddings.npy")
    paths_file = os.path.join(emb_dir, "image_paths.pkl")

    if not force and os.path.exists(emb_file) and os.path.exists(paths_file):
        existing = np.load(emb_file)
        print(f"Embeddings exist ({existing.shape[0]} images). Use --force to regenerate.")
        return

    image_paths = _find_images(frames_dir)
    if not image_paths:
        print(f"No images found in {frames_dir}")
        return

    device = _get_device()
    print(f"Model: {model_name}")
    print(f"Device: {device} | Images: {len(image_paths)} | Batch size: {batch_size}")

    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    valid_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        batch_valid = []

        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                batch_valid.append(p)
            except Exception as e:
                print(f"  Skip {p}: {e}")

        if not imgs:
            continue

        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            # SigLIP 2 returns an output object; extract the pooled tensor
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output

        all_embeddings.append(features.cpu().numpy())
        valid_paths.extend(batch_valid)

        del imgs, inputs, features
        if device == "cuda":
            torch.cuda.empty_cache()

    if not all_embeddings:
        print("No embeddings generated.")
        return

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # L2-normalize so dot product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    np.save(emb_file, embeddings)
    with open(paths_file, "wb") as f:
        pickle.dump(valid_paths, f)

    print(f"Saved {embeddings.shape[0]} embeddings ({embeddings.shape[1]}d) → {emb_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SigLIP 2 embeddings for frames")
    parser.add_argument("frames_dir", help="Directory containing frame images")
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate_embeddings(args.frames_dir, args.output, args.batch_size, args.model, args.force)