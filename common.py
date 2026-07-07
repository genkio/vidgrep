import os
import sqlite3
from pathlib import Path

os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # silence hub token nag on every run

import open_clip
import sqlite_vec
import torch
from sqlite_vec import serialize_float32

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
EMBED_DIM = 768
DEFAULT_DB = Path(__file__).parent / "index.db"


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
    db = sqlite3.connect(str(path))
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
    """)
    return db


def embed_text(model, tokenizer, device: str, query: str) -> torch.Tensor:
    with torch.no_grad():
        feat = model.encode_text(tokenizer([query]).to(device))
    return (feat / feat.norm(dim=-1, keepdim=True))[0].cpu()


def search_shots(db: sqlite3.Connection, query_vec: torch.Tensor, k: int) -> list[tuple[str, float, float, float]]:
    hits = db.execute(
        "SELECT rowid, distance FROM vec_shots WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_float32(query_vec.tolist()), k),
    ).fetchall()
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
