# lyric-to-image

Minimal pipeline: **video frames → SigLIP 2 embeddings → text lookup**.

Drop any videos into a folder, run one command, get a searchable image bank
you can query from TouchDesigner with a single lyric word or phrase.

## Quick Start

```bash
pip install -r requirements.txt
```

### One command — full pipeline

```bash
python run.py ./videos
```

This extracts frames from every video in `./videos/` (in parallel),
generates SigLIP 2 embeddings on your GPU, and saves everything to `./output/`.

### Start the lookup server

```bash
python server.py
```

### Query it

```bash
curl "http://localhost:8976/lookup?q=ramen"
# → {"query": "ramen", "video": "/abs/path/to/Naruto_Ramen.mp4", "timestamp": "01:26", "frame": "14dedb2f-01:26.jpg", "score": 0.145}

curl "http://localhost:8976/lookup?q=explosion&top=3"
# → {"query": "explosion", "results": [...]}
```

## Downloading Videos

Use the included helper to download from YouTube:

```bash
python utils/download_yt_videos.py "https://youtube.com/playlist?list=PLxxxxxx" ./videos
```

Or a `.txt` file with one URL per line:

```bash
python utils/download_yt_videos.py urls.txt ./videos
```

The video dir can contain **any** video files from any source:
YouTube downloads, screen recordings, stock footage, phone videos, etc.

Supported formats: `.mp4`, `.mkv`, `.webm`, `.avi`, `.mov`, `.m4v`, `.flv`

## Pipeline Options

```bash
python run.py ./videos --interval 1          # 1 frame/sec (more frames)
python run.py ./videos --interval 5          # 1 frame/5sec (fewer frames)
python run.py ./videos --head-skip 3         # skip first 3s (default: 5)
python run.py ./videos --tail-skip 15        # skip last 15s (default: 10)
python run.py ./videos --sim-threshold 0.95  # stricter duplicate filtering
python run.py ./videos --batch-size 64       # bigger batches for embedding
python run.py ./videos -j 4                  # limit to 4 parallel workers
python run.py ./videos --output ./my_out     # custom output dir
python run.py ./videos --force               # regenerate everything
```

### Frame Extraction

Frames are extracted every `--interval` seconds (default: 3), skipping the
first 5 seconds (`--head-skip`) and last 10 seconds (`--tail-skip`) of each
video to avoid intros/outros. Near-duplicate frames are filtered using
HSV histogram comparison (`--sim-threshold`).

Extraction runs in parallel using all available CPU cores by default.

### Embedding Model

Uses [SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch14-384)
(`google/siglip2-so400m-patch14-384`) for 1152-dimensional embeddings with
rich semantic understanding of composition, mood, style, and action.
Runs on MPS (Apple Silicon), CUDA, or CPU automatically.

## Frame Naming

Frames are named `{hash}-{MM:SS}.jpg` where the hash is derived from
the video's absolute path. A `manifest.json` maps each hash back to
its source video:

```
output/frames/
  a3f2b1c4-00:05.jpg
  a3f2b1c4-00:08.jpg
  a3f2b1c4-00:11.jpg
  b7e9d012-01:24.jpg
  ...
```

```json
// output/manifest.json
{
  "a3f2b1c4": "/Users/isaac/videos/nature_documentary.mp4",
  "b7e9d012": "/Volumes/External/stock/sunset.mp4"
}
```

## Output Structure

```
output/
├── manifest.json                    # hash → absolute video path
├── frames/                          # all extracted frames (flat)
│   ├── a3f2b1c4-00:05.jpg
│   ├── a3f2b1c4-00:08.jpg
│   └── ...
└── embeddings/
    ├── image_embeddings.npy         # (N, 1152) float32, L2-normalized
    └── image_paths.pkl              # list[str] of N image paths
```

## Server

### Start

```bash
python server.py                      # default: ./output on port 8976
python server.py --output ./my_out    # custom output dir
python server.py --port 9000          # custom port
```

The server loads SigLIP 2 and the embeddings once on startup, then serves
queries over HTTP. Each lookup is ~10ms after startup.

### Endpoints

```
GET /lookup?q=fire
→ {"query": "fire", "video": "/abs/path/to/video.mp4", "timestamp": "01:24", "frame": "...", "score": 0.14}

GET /lookup?q=ocean&top=3
→ {"query": "ocean", "results": [{"video": "...", "timestamp": "00:06", "score": 0.13}, ...]}

GET /health
→ {"status": "ok", "frames": 2083, "videos": 28}
```

## TouchDesigner Integration

### Runtime flow

1. `spotify-td` gives you the current lyric line
2. Extract a search term from the lyric (e.g. "walking through the rain" → "rain")
3. TD hits `server.py` with the search term
4. Server returns the source video path + timestamp
5. TD seeks the video to that timestamp (Movie File In TOP)

### Example TD Script

```python
import urllib.request, json

def lookup(word):
    url = f"http://localhost:8976/lookup?q={word}"
    resp = urllib.request.urlopen(url)
    data = json.loads(resp.read())
    return data["video"], data["timestamp"]
```

## Files

| File | Purpose |
|---|---|
| `run.py` | Master script — runs full build pipeline end to end |
| `extract_frames.py` | Video → frames with dedup, parallel processing (OpenCV) |
| `generate_embeddings.py` | Frames → SigLIP 2 embeddings (.npy + .pkl) |
| `server.py` | Lookup server — text in, video path + timestamp out |
| `utils/download_yt_videos.py` | Download YouTube videos/playlists via yt-dlp |