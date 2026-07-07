import argparse
import sys
from pathlib import Path

import torch
from sqlite_vec import serialize_float32

from common import DEFAULT_DB, fmt_time, get_device, load_clip, open_db


def main() -> None:
    ap = argparse.ArgumentParser(description="Natural-language search over indexed videos.")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=10, help="number of results")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()

    db = open_db(args.db)
    if db.execute("SELECT count(*) FROM shots").fetchone()[0] == 0:
        sys.exit("index is empty, run index.py first")

    device = get_device()
    model, _, tokenizer = load_clip(device)
    with torch.no_grad():
        feat = model.encode_text(tokenizer([args.query]).to(device))
    feat = (feat / feat.norm(dim=-1, keepdim=True))[0].cpu()

    hits = db.execute(
        "SELECT rowid, distance FROM vec_shots WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_float32(feat.tolist()), args.k),
    ).fetchall()

    for rank, (rowid, dist) in enumerate(hits, 1):
        path, start, end = db.execute(
            "SELECT v.path, s.start_s, s.end_s FROM shots s"
            " JOIN videos v ON v.id = s.video_id WHERE s.id = ?",
            (rowid,),
        ).fetchone()
        score = 1 - dist * dist / 2  # cosine from L2 on unit vectors
        print(f"{rank:2d}  {score:.3f}  {Path(path).name}  {fmt_time(start)}-{fmt_time(end)}")
        print(f'    mpv --start={int(start)} "{path}"')


if __name__ == "__main__":
    main()
