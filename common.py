import sqlite3
from pathlib import Path

import open_clip
import sqlite_vec
import torch

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512
DEFAULT_DB = Path(__file__).parent / "index.db"


def get_device() -> str:
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


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
