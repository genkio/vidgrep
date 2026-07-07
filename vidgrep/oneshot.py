import argparse
import shutil
import sys
from pathlib import Path

from vidgrep.common import DEFAULT_DB, MODEL_NAME, embed_text, get_device, load_clip, open_db, search_shots
from vidgrep.cut import export_clips
from vidgrep.index import ensure_indexed, find_videos


def parse_jobs(spec: list[str]) -> list[tuple[str, Path]]:
    if len(spec) == 1:
        return [(spec[0], Path("output"))]
    if len(spec) % 2 != 0:
        sys.exit('descriptions and output folders must come in pairs: "desc a" ./a "desc b" ./b')
    return [(spec[i], Path(spec[i + 1])) for i in range(0, len(spec), 2)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Index video(s) and cut clips matching each description, one video at a time.",
        epilog='multiple descriptions: vidgrep oneshot . "desc a" ./a "desc b" ./b',
    )
    ap.add_argument("path", type=Path, help="video file, or folder searched recursively")
    ap.add_argument(
        "spec",
        nargs="+",
        metavar="DESC [DIR]",
        help="description of the clips you want, followed by its output folder"
        " (a single description defaults to ./output)",
    )
    ap.add_argument("-k", type=int, default=10, help="number of clips per video per description")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--pad", type=float, default=0.5, help="seconds added before/after each clip")
    args = ap.parse_args()

    jobs = parse_jobs(args.spec)

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found, install with: brew install ffmpeg")

    videos = find_videos(args.path.expanduser())
    if not videos:
        sys.exit(f"no video files found under {args.path}")

    device = get_device()
    print(f"{len(videos)} video(s), {len(jobs)} description(s), {MODEL_NAME} on {device}")
    model, preprocess, tokenizer = load_clip(device)
    db = open_db(args.db)
    queries = [(embed_text(model, tokenizer, device, desc), out) for desc, out in jobs]

    # cut per video so first clips appear without waiting for the full run
    failed = []
    for i, path in enumerate(videos, 1):
        # keep going on bad files: one corrupt video must not kill an overnight run
        try:
            ensure_indexed(db, model, preprocess, device, path, f"[{i}/{len(videos)}] {path.name}")
            row = db.execute("SELECT id FROM videos WHERE path = ?", (str(path),)).fetchone()
            if row is None:
                continue
            for query_vec, out in queries:
                results = search_shots(db, query_vec, args.k, [row[0]])
                export_clips(results, out, args.pad)
        except Exception as e:
            failed.append(path.name)
            print(f"[{i}/{len(videos)}] {path.name}  FAILED: {e}")
    if failed:
        print(f"\n{len(failed)} video(s) failed, re-run the same command to retry them:")
        for name in failed:
            print(f"  {name}")
