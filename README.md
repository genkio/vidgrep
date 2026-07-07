# vidgrep

Natural-language search over local video files. Fully local: no cloud, no subtitles, no audio. Frames only.

## How it works

1. `index.py` splits each video into shots (scene-cut detection), grabs one keyframe per shot, embeds it with CLIP (image encoder), stores vector + `{path, start, end}` in `index.db` (sqlite-vec).
2. `search.py` embeds your phrase with CLIP (text encoder, same vector space), runs nearest-neighbor over the stored vectors, prints timestamps.

No training anywhere. CLIP arrives pre-trained; first run downloads weights (~600 MB) to `~/.cache`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Optional: `brew install mpv` to jump straight to results.

```bash
uv sync
```

## Usage

```bash
# index one file, or a folder (recursive) - slow, one-time, resumable
uv run index.py ~/Videos/

# search - instant
uv run search.py "two people kissing in a coffee shop"
uv run search.py "a dog running on a beach" -k 20
```

Output: score (cosine, ~0.3 = strong hit, rank matters not the number), file, time range, ready-to-paste mpv command.

Re-running `index.py` skips already-indexed files (re-indexes if the file changed). Delete `index.db` to start over.

## Tuning

Constants at the top of `index.py` / `common.py`:

- `MODEL_NAME` / `PRETRAINED`: `ViT-B-32` is fast; `ViT-L-14` (`EMBED_DIM = 768`) is slower + better. Changing models requires re-indexing.
- `MAX_UNIT_S` / `SPLIT_STEP_S`: long shots get one sample per 10 s.
- `FALLBACK_WINDOW_S`: window size for footage with no scene cuts (e.g. GoPro).

## Roadmap

- face tagging (InsightFace) -> filter by who is in the shot
- VLM re-rank of top candidates (Qwen-VL via mlx)
- web UI over the same search fn
