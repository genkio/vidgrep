import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from scenedetect import ContentDetector, detect
from sqlite_vec import serialize_float32

from common import DEFAULT_DB, MODEL_NAME, get_device, load_clip, open_db

VIDEO_EXTS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".webm", ".wmv"}
MAX_UNIT_S = 20.0
SPLIT_STEP_S = 10.0
FALLBACK_WINDOW_S = 10.0
BATCH_SIZE = 64


def find_videos(root: Path) -> list[Path]:
    if root.is_file():
        return [root.resolve()]
    return sorted(
        p.resolve()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def video_duration_s(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0.0


def shot_spans(path: Path) -> list[tuple[float, float]]:
    scenes = detect(str(path), ContentDetector())
    spans = [(start.seconds, end.seconds) for start, end in scenes]
    if spans:
        return spans
    # no cuts detected (single-shot footage): fixed windows keep it searchable
    duration = video_duration_s(path)
    spans, t = [], 0.0
    while t < duration:
        spans.append((t, min(t + FALLBACK_WINDOW_S, duration)))
        t += FALLBACK_WINDOW_S
    return spans


def sample_units(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    # split long shots so a slow scene isn't reduced to one frame
    units = []
    for start, end in spans:
        if end - start <= MAX_UNIT_S:
            units.append((start, end))
        else:
            t = start
            while t < end:
                units.append((t, min(t + SPLIT_STEP_S, end)))
                t += SPLIT_STEP_S
    return units


def grab_frame(cap: cv2.VideoCapture, t: float) -> Image.Image | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def index_video(db, model, preprocess, device: str, path: Path) -> int:
    units = sample_units(shot_spans(path))
    cap = cv2.VideoCapture(str(path))
    metas: list[tuple[float, float]] = []
    tensors: list[torch.Tensor] = []
    vecs: list[torch.Tensor] = []

    def flush() -> None:
        if not tensors:
            return
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.extend(feats.cpu())
        tensors.clear()

    for start, end in units:
        img = grab_frame(cap, (start + end) / 2)
        if img is None:
            continue
        tensors.append(preprocess(img))
        metas.append((start, end))
        if len(tensors) >= BATCH_SIZE:
            flush()
    flush()
    cap.release()

    cur = db.cursor()
    cur.execute(
        "INSERT INTO videos(path, mtime, shot_count) VALUES(?, ?, ?)",
        (str(path), path.stat().st_mtime, len(metas)),
    )
    video_id = cur.lastrowid
    for (start, end), vec in zip(metas, vecs):
        cur.execute(
            "INSERT INTO shots(video_id, start_s, end_s) VALUES(?, ?, ?)",
            (video_id, start, end),
        )
        cur.execute(
            "INSERT INTO vec_shots(rowid, embedding) VALUES(?, ?)",
            (cur.lastrowid, serialize_float32(vec.tolist())),
        )
    db.commit()
    return len(metas)


def purge_video(db, video_id: int) -> None:
    shot_ids = [r[0] for r in db.execute("SELECT id FROM shots WHERE video_id = ?", (video_id,))]
    db.executemany("DELETE FROM vec_shots WHERE rowid = ?", [(i,) for i in shot_ids])
    db.execute("DELETE FROM shots WHERE video_id = ?", (video_id,))
    db.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    db.commit()


def index_all(db, model, preprocess, device: str, videos: list[Path]) -> None:
    for i, path in enumerate(videos, 1):
        label = f"[{i}/{len(videos)}] {path.name}"
        row = db.execute("SELECT id, mtime FROM videos WHERE path = ?", (str(path),)).fetchone()
        if row:
            if row[1] == path.stat().st_mtime:
                print(f"{label}  already indexed, skipping")
                continue
            purge_video(db, row[0])
        t0 = time.time()
        n = index_video(db, model, preprocess, device, path)
        print(f"{label}  {n} shots  {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Index video file(s) for natural-language search.")
    ap.add_argument("path", type=Path, help="video file, or folder searched recursively")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()

    videos = find_videos(args.path.expanduser())
    if not videos:
        sys.exit(f"no video files found under {args.path}")

    device = get_device()
    print(f"{len(videos)} video(s), {MODEL_NAME} on {device}")
    model, preprocess, _ = load_clip(device)
    db = open_db(args.db)
    index_all(db, model, preprocess, device, videos)


if __name__ == "__main__":
    main()
