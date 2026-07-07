import argparse
import sys
from pathlib import Path

from common import DEFAULT_DB, embed_text, fmt_time, get_device, load_clip, open_db, search_shots


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
    results = search_shots(db, embed_text(model, tokenizer, device, args.query), args.k)

    for rank, (path, start, end, score) in enumerate(results, 1):
        print(f"{rank:2d}  {score:.3f}  {Path(path).name}  {fmt_time(start)}-{fmt_time(end)}")
        print(f'    mpv --start={int(start)} "{path}"')


if __name__ == "__main__":
    main()
