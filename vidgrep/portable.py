import json
from pathlib import Path

import numpy as np

from vidgrep.common import VIDEO_EXTS

ONNX_FILE = "text_encoder.onnx"
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "encoder.json"


class Encoder:
    """Torch-free CLIP text encoder: tokenizer.json + ONNX text tower run on CPU."""

    def __init__(self, directory: Path | str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(directory).expanduser()
        self.cfg = json.loads((d / CONFIG_FILE).read_text())
        self.tok = Tokenizer.from_file(str(d / TOKENIZER_FILE))
        self.tok.no_padding()
        self.tok.no_truncation()
        self.sess = ort.InferenceSession(
            str(d / ONNX_FILE), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name

    # must reproduce open_clip's token wrapping exactly (verified at export time);
    # a silent mismatch would degrade every query
    def _tokens(self, text: str) -> np.ndarray:
        ctx = self.cfg["context_length"]
        sot, eot = self.cfg["sot_id"], self.cfg["eot_id"]
        ids = [sot, *self.tok.encode(text, add_special_tokens=False).ids, eot][:ctx]
        ids[-1] = eot
        ids += [0] * (ctx - len(ids))
        return np.array([ids], dtype=np.int64)

    def embed(self, text: str) -> np.ndarray:
        (feat,) = self.sess.run(None, {self.input_name: self._tokens(text)})
        v = feat[0].astype(np.float32)
        return v / np.linalg.norm(v)


def local_video_map(root: Path) -> dict[str, str]:
    root = root.expanduser()
    paths = [root] if root.is_file() else root.rglob("*")
    out: dict[str, str] = {}
    for p in paths:
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out[p.name] = str(p.resolve())  # basename -> local abspath
    return out


def remap_paths(results, local_map: dict[str, str]):
    remapped = []
    for path, start, end, score in results:
        local = local_map.get(Path(path).name, path)
        remapped.append((local, start, end, score))
    return remapped
