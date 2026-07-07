import argparse
import shutil
import sys
from pathlib import Path

from common import DEFAULT_DB, MODEL_NAME, embed_text, get_device, load_clip, open_db, search_shots
from cut import export_clips
from index import ensure_indexed, find_videos


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Index video(s) and cut clips matching a description, one video at a time."
    )
    ap.add_argument("path", type=Path, help="video file, or folder searched recursively")
    ap.add_argument("query", help="description of the clips you want")
    ap.add_argument("-k", type=int, default=10, help="number of clips per video")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=Path("output"), help="clip output folder")
    ap.add_argument("--pad", type=float, default=0.5, help="seconds added before/after each clip")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found, install with: brew install ffmpeg")

    videos = find_videos(args.path.expanduser())
    if not videos:
        sys.exit(f"no video files found under {args.path}")

    device = get_device()
    print(f"{len(videos)} video(s), {MODEL_NAME} on {device}")
    model, preprocess, tokenizer = load_clip(device)
    db = open_db(args.db)
    query_vec = embed_text(model, tokenizer, device, args.query)

    # cut per video so first clips appear without waiting for the full run
    for i, path in enumerate(videos, 1):
        ensure_indexed(db, model, preprocess, device, path, f"[{i}/{len(videos)}] {path.name}")
        row = db.execute("SELECT id FROM videos WHERE path = ?", (str(path),)).fetchone()
        if row is None:
            continue
        results = search_shots(db, query_vec, args.k, [row[0]])
        export_clips(results, args.out, args.pad)


if __name__ == "__main__":
    main()
