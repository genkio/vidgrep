import argparse
import shutil
import sys
from pathlib import Path

from common import DEFAULT_DB, MODEL_NAME, embed_text, get_device, load_clip, open_db, search_shots
from cut import export_clips
from index import find_videos, index_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Index video(s) and cut clips matching a description, in one go.")
    ap.add_argument("path", type=Path, help="video file, or folder searched recursively")
    ap.add_argument("query", help="description of the clips you want")
    ap.add_argument("-k", type=int, default=10, help="number of clips")
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

    index_all(db, model, preprocess, device, videos)

    placeholders = ",".join("?" * len(videos))
    video_ids = [
        r[0]
        for r in db.execute(
            f"SELECT id FROM videos WHERE path IN ({placeholders})",
            [str(p) for p in videos],
        )
    ]
    # scope to the given path: a shared index may hold unrelated videos
    results = search_shots(db, embed_text(model, tokenizer, device, args.query), args.k, video_ids)
    export_clips(results, args.out, args.pad)


if __name__ == "__main__":
    main()
