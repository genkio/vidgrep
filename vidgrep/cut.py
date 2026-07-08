import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from vidgrep.common import DEFAULT_DB, fmt_time, open_db, parse_jobs, search_shots


def export_clips(results: list[tuple[str, float, float, float]], out: Path, pad: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for rank, (path, start, end, score) in enumerate(results, 1):
        src = Path(path)
        if not src.exists():
            print(f"{rank:2d}  source moved or deleted, skipped: {src}")
            continue
        cut_start = max(0.0, start - pad)
        duration = end + pad - cut_start
        clip = out / f"{rank:02d}_{src.stem}_{int(start)}s.mp4"
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Search indexed videos and cut the results into clips.",
        epilog='multiple descriptions: vidgrep cut "desc a" ./a "desc b" ./b',
    )
    ap.add_argument(
        "spec",
        nargs="+",
        metavar="DESC [DIR]",
        help="description of the clips you want, followed by its output folder"
        " (a single description defaults to ./output)",
    )
    ap.add_argument("-k", type=int, default=10, help="number of clips per description")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--pad", type=float, default=0.5, help="seconds added before/after each clip")
    ap.add_argument(
        "--encoder",
        type=Path,
        help="path to an exported encoder bundle; enables cutting without PyTorch"
        " (see: vidgrep export-encoder)",
    )
    ap.add_argument(
        "--videos",
        type=Path,
        help="remap indexed source paths to local files under this path, matched by filename"
        " (for indexes built on another machine)",
    )
    args = ap.parse_args()

    jobs = parse_jobs(args.spec)

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found, install with: brew install ffmpeg")

    db = open_db(args.db)
    if db.execute("SELECT count(*) FROM shots").fetchone()[0] == 0:
        sys.exit("index is empty, run vidgrep index first")

    embed = _make_embedder(args.encoder)
    local_map = None
    if args.videos:
        from vidgrep.portable import local_video_map, remap_paths

        local_map = local_video_map(args.videos)

    for desc, out in jobs:
        if len(jobs) > 1:
            print(f"== {desc} -> {out}/")
        results = search_shots(db, embed(desc), args.k)
        if local_map is not None:
            results = remap_paths(results, local_map)
        export_clips(results, out, args.pad)


def _make_embedder(encoder_dir: Path | None):
    if encoder_dir is not None:
        from vidgrep.portable import Encoder

        enc = Encoder(encoder_dir)
        return enc.embed
    # torch path, imported lazily so the --encoder route needs no PyTorch
    try:
        from vidgrep.common import embed_text, get_device, load_clip

        device = get_device()
    except ModuleNotFoundError as e:
        if e.name in ("torch", "open_clip"):
            sys.exit(
                "PyTorch isn't installed here. Cut without it by passing an exported"
                " encoder: --encoder <bundle> (build one with: vidgrep export-encoder)."
            )
        raise
    model, _, tokenizer = load_clip(device)
    return lambda desc: embed_text(model, tokenizer, device, desc)
