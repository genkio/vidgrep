# vidgrep

Natural-language search over local video files. Fully local: no cloud, no subtitles, no audio. Frames only.

## How it works

1. `index.py` splits each video into shots (scene-cut detection), grabs one keyframe per shot, embeds it with CLIP (image encoder), stores vector + `{path, start, end}` in `index.db` (sqlite-vec).
2. `search.py` embeds your phrase with CLIP (text encoder, same vector space), runs nearest-neighbor over the stored vectors, prints timestamps.
3. `cut.py` runs the same search, then ffmpeg cuts each hit into a standalone clip.

No training anywhere. CLIP arrives pre-trained; first run downloads weights (~1.7 GB for the default ViT-L-14) to `~/.cache`.

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

# cut top results into ./output/*.mp4 (needs ffmpeg: brew install ffmpeg)
uv run cut.py "a dog running on a beach" -k 5 --pad 1

```

Output: score (cosine, ~0.3 = strong hit, rank matters not the number), file, time range, ready-to-paste mpv command.

Re-running `index.py` skips already-indexed files (re-indexes if the file changed).

## FAQ

**What does `-k` mean?** Number of results returned (top-K nearest matches). Default 10.

**Can I search in other languages?** No, English only: the default model was trained on English captions. For multilingual queries set `MODEL_NAME = "xlm-roberta-base-ViT-B-32"`, `PRETRAINED = "laion5b_s13b_b90k"` in `common.py` and re-index.

**How do I start over, or index a different set of videos?** The whole index is one file. Delete `index.db` to start fresh, or keep collections side by side with `--db`:

```bash
uv run index.py ~/Videos/trips/ --db trips.db
uv run search.py "sunset over water" --db trips.db
```

**Where do clips go?** `cut.py` writes to `./output` in the current directory (`--out` to change), padding each clip by 0.5 s of context (`--pad` to change).

## Tuning

Constants at the top of `index.py` / `common.py`:

- `MODEL_NAME` / `PRETRAINED`: `ViT-L-14` (default) is accurate; `ViT-B-32` (`EMBED_DIM = 512`) indexes ~4x faster, weaker results. Changing models requires re-indexing (delete `index.db`).
- `MAX_UNIT_S` / `SPLIT_STEP_S`: long shots get one sample per 10 s.
- `FALLBACK_WINDOW_S`: window size for footage with no scene cuts (e.g. GoPro).

## Roadmap

- face tagging (InsightFace) -> filter by who is in the shot
- VLM re-rank of top candidates (Qwen-VL via mlx)
- web UI over the same search fn
