# lyric-to-image

Minimal pipeline: **video frames → CLIP embeddings → word lookup**.

Drop any videos into a folder, run one command, get a searchable image bank
you can query from TouchDesigner with a single lyric word.

## Quick Start

```bash
pip install -r requirements.txt
```

### One command — full pipeline

```bash
python run.py ./my_videos
```

That's it. It extracts frames from every video in `./my_videos/`, generates
CLIP embeddings, and saves everything to `./output/`.

### Options

```bash
python run.py ./my_videos --interval 1        # 1 frame/sec (more frames)
python run.py ./my_videos --interval 5        # 1 frame/5sec (fewer frames)
python run.py ./my_videos --tail-skip 15      # skip last 15s of each video
python run.py ./my_videos --sim-threshold 0.95 # stricter duplicate filtering
python run.py ./my_videos --batch-size 32     # bigger batches (more VRAM)
python run.py ./my_videos --output ./my_out   # custom output dir
python run.py ./my_videos --force             # regenerate everything
```

## Video Sources

The `video_dir` can contain **any** video files from any source:
YouTube downloads, screen recordings, stock footage, phone videos, etc.

Supported formats: `.mp4`, `.mkv`, `.webm`, `.avi`, `.mov`, `.m4v`, `.flv`

If you want to download from a YouTube playlist, use
[yt-dlp](https://github.com/yt-dlp/yt-dlp) separately:

```bash
yt-dlp --format "bestvideo[height<=720]+bestaudio/best" \
       --merge-output-format mp4 \
       --restrict-filenames \
       -o "./my_videos/%(title)s.%(ext)s" \
       "https://youtube.com/playlist?list=PLxxxxxx"
```

## Frame Naming

Frames are named `{hash}-{MM:SS}.jpg` where the hash is derived from
the video's absolute path. A `manifest.json` maps each hash back to
its source video:

```
output/frames/
  a3f2b1c4-00:03.jpg
  a3f2b1c4-00:06.jpg
  a3f2b1c4-00:09.jpg
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

This keeps filenames short and source-agnostic — videos can live anywhere
on disk and frames are always traceable. The last 10 seconds of each video
are skipped by default (`--tail-skip`) to avoid credits/outros.

## Output Structure

```
output/
├── manifest.json                    # hash → absolute video path
├── frames/                          # all extracted frames (flat)
│   ├── a3f2b1c4-00:03.jpg
│   ├── a3f2b1c4-00:06.jpg
│   └── ...
└── embeddings/
    ├── image_embeddings.npy         # (N, 512) float32, L2-normalized
    └── image_paths.pkl              # list[str] of N image paths
```

## TouchDesigner Integration

This repo produces the offline assets and a lightweight lookup server.

### Start the server

```bash
python server.py                      # default: ./output on port 8976
python server.py --output ./my_out    # custom output dir
python server.py --port 9000          # custom port
```

The server loads CLIP and the embeddings once on startup, then serves
queries over HTTP. Each lookup is ~10ms.

### Query from TD

From a TD Web Client DAT or Script DAT, just hit the endpoint:

```
GET http://localhost:8976/lookup?q=fire
→ {"query": "fire", "video": "/abs/path/to/nature.mp4", "timestamp": "01:24"}

GET http://localhost:8976/lookup?q=ocean&top=3
→ {"query": "ocean", "results": [{"video": "...", "timestamp": "00:06"}, ...]}
```

```python
# Example TD Script DAT
import urllib.request, json

def lookup(word):
    url = f"http://localhost:8976/lookup?q={word}"
    resp = urllib.request.urlopen(url)
    data = json.loads(resp.read())
    return data["video"], data["timestamp"]
```

### Runtime flow

1. `spotify-td` gives you the current lyric line
2. Claude Code turns it into a search term (e.g. "walking through the rain" → "rain")
3. TD hits `server.py` with the search term
4. Server returns the source video path + timestamp
5. TD seeks the video to that timestamp (Movie File In TOP)

### Health check

```
GET http://localhost:8976/health
→ {"status": "ok", "frames": 2847, "videos": 12}
```

## Files

| File | Purpose |
|---|---|
| `run.py` | Master script — runs full build pipeline end to end |
| `extract_frames.py` | Video → frames with dedup (OpenCV) |
| `generate_embeddings.py` | Frames → CLIP embeddings (.npy + .pkl) |
| `server.py` | Lookup server — word in, video path + timestamp out (run alongside TD) |