# vidgrep

Natural-language search over local video files. Fully local: no cloud, no subtitles, no audio. Frames only.

```bash
vidgrep oneshot ~/Videos "a dog jumping into a lake" -k 5
```

## Install

```bash
brew install genkio/tap/vidgrep
```

First run downloads the model weights (~2 GB) to `~/.cache`. Optional: `brew install mpv` to jump straight to search results.

## Usage

```bash
# index + cut in one go, one video at a time: clips appear as each video finishes
# (-k = clips per video; for the global top-k afterwards, run vidgrep cut - index is already built)
vidgrep oneshot ~/Videos/trips/ "sunset over water" -k 5

# several descriptions in one pass, each with its own output folder
# (indexing dominates the cost, extra descriptions are nearly free)
vidgrep oneshot ~/Videos "a dog jumping into a lake" ./dog "sunset over water" ./sunset

# long unattended runs: bad files are skipped and listed at the end, re-run to retry;
# caffeinate keeps the mac awake
caffeinate -i vidgrep oneshot ~/Videos "a dog jumping into a lake" ./dog

# index one file, or a folder (recursive) - slow, one-time, resumable
vidgrep index ~/Videos/

# search - instant
vidgrep search "two people kissing in a coffee shop"
vidgrep search "a dog running on a beach" -k 20

# cut top results into ./output/*.mp4 (or several descriptions, each with its own folder)
vidgrep cut "a dog running on a beach" -k 5 --pad 1
vidgrep cut "a dog running on a beach" ./dog "sunset over water" ./sunset
```

Search output: score (cosine, ~0.3 = strong hit, rank matters not the number), file, time range, ready-to-paste mpv command.

Re-running `vidgrep index` skips already-indexed files (re-indexes if the file changed).

## How it works

1. `index` splits each video into shots (scene-cut detection), grabs one keyframe per shot, embeds it with CLIP (image encoder), stores vector + `{path, start, end}` in `~/.vidgrep/index.db` (sqlite-vec).
2. `search` embeds your phrase with CLIP (text encoder, same vector space), runs nearest-neighbor over the stored vectors, prints timestamps.
3. `cut` runs the same search, then ffmpeg cuts each hit into a standalone clip.

No training anywhere. CLIP arrives pre-trained.

## FAQ

**What does `-k` mean?** Number of results returned (top-K nearest matches). Default 10. For `oneshot` it's per video.

**Can I search in other languages?** No, English only: the default model was trained on English captions. For multilingual queries set `MODEL_NAME = "xlm-roberta-base-ViT-B-32"`, `PRETRAINED = "laion5b_s13b_b90k"` in `vidgrep/common.py` and re-index.

**How do I start over, or index a different set of videos?** The whole index is one file. Delete `~/.vidgrep/index.db` to start fresh, or keep collections side by side with `--db`:

```bash
vidgrep index ~/Videos/trips/ --db trips.db
vidgrep search "sunset over water" --db trips.db
```

**Where do clips go?** `./output` in the current directory by default, padded by 0.5 s of context (`--pad` to change). Both `cut` and `oneshot` take an output folder right after each description.

## Tuning

Constants at the top of `vidgrep/index.py` / `vidgrep/common.py`:

- `MODEL_NAME` / `PRETRAINED` / `EMBED_DIM`: the CLIP model. Any [open_clip](https://github.com/mlfoundations/open_clip) model works, including `hf-hub:<repo>` checkpoints; a plain `ViT-B-32` / `laion2b_s34b_b79k` / `512` indexes fastest with weaker results. Each index is locked to the model that built it; changing models requires a fresh re-index (delete `~/.vidgrep/index.db` or use a separate `--db`).
- `MAX_UNIT_S` / `SPLIT_STEP_S`: long shots get one sample per 10 s.
- `FALLBACK_WINDOW_S`: window size for footage with no scene cuts (e.g. GoPro).

## Development

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
git clone https://github.com/genkio/vidgrep && cd vidgrep
uv sync
uv run vidgrep search "..."
```

## Roadmap

- face tagging (InsightFace) -> filter by who is in the shot
- VLM re-rank of top candidates (Qwen-VL via mlx)
- web UI over the same search fn

## License

MIT
