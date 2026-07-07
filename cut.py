import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_DB, embed_text, fmt_time, get_device, load_clip, open_db, search_shots


def main() -> None:
    ap = argparse.ArgumentParser(description="Search indexed videos and cut the results into clips.")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=10, help="number of clips")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=Path("output"), help="clip output folder")
    ap.add_argument("--pad", type=float, default=0.5, help="seconds added before/after each clip")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found, install with: brew install ffmpeg")

    db = open_db(args.db)
    if db.execute("SELECT count(*) FROM shots").fetchone()[0] == 0:
        sys.exit("index is empty, run index.py first")

    device = get_device()
    model, _, tokenizer = load_clip(device)
    results = search_shots(db, embed_text(model, tokenizer, device, args.query), args.k)

    args.out.mkdir(parents=True, exist_ok=True)
    for rank, (path, start, end, score) in enumerate(results, 1):
        src = Path(path)
        if not src.exists():
            print(f"{rank:2d}  source moved or deleted, skipped: {src}")
            continue
        cut_start = max(0.0, start - args.pad)
        duration = end + args.pad - cut_start
        clip = args.out / f"{rank:02d}_{src.stem}_{int(start)}s.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{cut_start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
            # re-encode: stream copy would snap to keyframes and miss the shot start
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac",
            str(clip),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"{rank:2d}  ffmpeg failed on {src.name}, skipped")
            continue
        print(f"{rank:2d}  {score:.3f}  {clip}  ({fmt_time(start)}-{fmt_time(end)})")


if __name__ == "__main__":
    main()
