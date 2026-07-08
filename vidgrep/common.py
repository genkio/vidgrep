import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # silence hub token nag on every run

import open_clip
import sqlite_vec
import torch
from sqlite_vec import serialize_float32

MODEL_NAME = "hf-hub:woweenie/open-clip-vit-h-nsfw-finetune"
PRETRAINED = None  # hf-hub repo bundles its own weights, no separate pretrained tag
EMBED_DIM = 1024
DEFAULT_DB = Path.home() / ".vidgrep" / "index.db"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_clip(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model.to(device).eval(), preprocess, tokenizer


def open_db(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(p))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.executescript(f"""
        CREATE TABLE IF NOT EXISTS videos(
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            mtime REAL NOT NULL,
            shot_count INTEGER
        );
        CREATE TABLE IF NOT EXISTS shots(
            id INTEGER PRIMARY KEY,
            video_id INTEGER NOT NULL REFERENCES videos(id),
            start_s REAL NOT NULL,
            end_s REAL NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_shots USING vec0(
            embedding float[{EMBED_DIM}]
        );
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
    """)
    _check_model(db, p)
    return db


# vectors from different models aren't comparable and differ in dimension, so a db is
# locked to the model that built it - guard turns a cryptic dim mismatch into a clear message
def _check_model(db: sqlite3.Connection, path: Path) -> None:
    current = f"{MODEL_NAME}|{EMBED_DIM}"
    row = db.execute("SELECT value FROM meta WHERE key = 'model'").fetchone()
    if row is None:
        if db.execute("SELECT count(*) FROM shots").fetchone()[0] > 0:
            sys.exit(f"{path} was built by an older vidgrep; delete it and re-index.")
        db.execute("INSERT INTO meta(key, value) VALUES('model', ?)", (current,))
        db.commit()
    elif row[0] != current:
        sys.exit(
            f"{path} was built with {row[0]}, active model is {current}.\n"
            "delete it and re-index, or use --db to keep a separate index per model."
        )


def parse_jobs(spec: list[str]) -> list[tuple[str, Path]]:
    if len(spec) == 1:
        return [(spec[0], Path("output"))]
    if len(spec) % 2 != 0:
        sys.exit('descriptions and output folders must come in pairs: "desc a" ./a "desc b" ./b')
    return [(spec[i], Path(spec[i + 1])) for i in range(0, len(spec), 2)]


def embed_text(model, tokenizer, device: str, query: str) -> torch.Tensor:
    with torch.no_grad():
        feat = model.encode_text(tokenizer([query]).to(device))
    return (feat / feat.norm(dim=-1, keepdim=True))[0].cpu()


def search_shots(
    db: sqlite3.Connection,
    query_vec: torch.Tensor,
    k: int,
    video_ids: list[int] | None = None,
) -> list[tuple[str, float, float, float]]:
    sql = "SELECT rowid, distance FROM vec_shots WHERE embedding MATCH ? AND k = ?"
    params: list = [serialize_float32(query_vec.tolist()), k]
    if video_ids:
        placeholders = ",".join("?" * len(video_ids))
        sql += f" AND rowid IN (SELECT id FROM shots WHERE video_id IN ({placeholders}))"
        params.extend(video_ids)
    hits = db.execute(sql + " ORDER BY distance", params).fetchall()
    results = []
    for rowid, dist in hits:
        path, start, end = db.execute(
            "SELECT v.path, s.start_s, s.end_s FROM shots s"
            " JOIN videos v ON v.id = s.video_id WHERE s.id = ?",
            (rowid,),
        ).fetchone()
        score = 1 - dist * dist / 2  # cosine from L2 on unit vectors
        results.append((path, start, end, score))
    return results


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
