import argparse
import json
import shutil
from pathlib import Path

from vidgrep.common import EMBED_DIM, MODEL_NAME, load_clip
from vidgrep.portable import CONFIG_FILE, ONNX_FILE, TOKENIZER_FILE, Encoder

PROBES = [
    "a woman in a wedding dress",
    "two people kissing in a coffee shop",
    "a dog running on a beach at sunset",
    "",
]
DEFAULT_OUT = Path.home() / ".vidgrep" / "encoder"


def _tokenizer_json() -> Path:
    from huggingface_hub import hf_hub_download

    repo = MODEL_NAME.split("hf-hub:", 1)[-1]
    return Path(hf_hub_download(repo, TOKENIZER_FILE))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export the text encoder to a portable ONNX bundle for torch-free cutting."
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="bundle output directory")
    args = ap.parse_args()

    import torch
    from torch import nn

    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_NAME} ...")
    model, _, tokenizer = load_clip("cpu")
    model = model.float().eval()

    ref = tokenizer(PROBES)  # [N, context_length] int64, the ground truth
    context_length = ref.shape[1]
    nonzero = ref[0][ref[0] != 0]
    sot_id, eot_id = int(nonzero[0]), int(nonzero[-1])

    class TextTower(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tokens):
            return self.m.encode_text(tokens)

    print("exporting text tower to ONNX ...")
    torch.onnx.export(
        TextTower(model),
        (ref[:1],),
        str(out / ONNX_FILE),
        input_names=["tokens"],
        output_names=["features"],
        dynamic_axes={"tokens": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
    )

    shutil.copyfile(_tokenizer_json(), out / TOKENIZER_FILE)
    (out / CONFIG_FILE).write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "embed_dim": EMBED_DIM,
                "context_length": context_length,
                "sot_id": sot_id,
                "eot_id": eot_id,
            },
            indent=2,
        )
    )

    print("verifying parity vs open_clip ...")
    enc = Encoder(out)
    with torch.no_grad():
        for text in PROBES:
            ids = enc._tokens(text)[0].tolist()
            want = ref[PROBES.index(text)].tolist()
            assert ids == want, f"tokenizer mismatch on {text!r}:\n {ids}\n {want}"
            feat = model.encode_text(tokenizer([text]))
            ref_vec = (feat / feat.norm(dim=-1, keepdim=True))[0].numpy()
            cos = float(enc.embed(text) @ ref_vec)
            assert cos > 0.999, f"embedding mismatch on {text!r}: cos={cos:.4f}"
    size_mb = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e6
    print(f"OK  bundle at {out}  ({size_mb:.0f} MB)  copy the whole folder + index.db to cut elsewhere")


if __name__ == "__main__":
    main()
