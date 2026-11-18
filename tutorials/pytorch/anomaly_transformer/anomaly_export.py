"""anomaly_export.py — export AnomalyTransformer checkpoint to fp32 TFLite.

Usage:
    python anomaly_export.py                        # SMD → exports/SMD_fp32.tflite
    python anomaly_export.py --dataset MSL
"""

import argparse, os, sys
import torch
import litert_torch

sys.path.insert(0, os.path.dirname(__file__))
from model.AnomalyTransformer import AnomalyTransformer

PRESETS = {
    "SMD": dict(win_size=100, enc_in=38, c_out=38),
    "MSL": dict(win_size=100, enc_in=55, c_out=55),
    "SMAP": dict(win_size=100, enc_in=25, c_out=25),
    "PSM": dict(win_size=100, enc_in=25, c_out=25),
}


def export(dataset: str, cfg: dict, ckpt_dir: str, out_dir: str) -> str:
    model = AnomalyTransformer(
        win_size=cfg["win_size"],
        enc_in=cfg["enc_in"],
        c_out=cfg["c_out"],
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.0,
        activation="gelu",
        output_attention=True,
    )
    ckpt = os.path.join(ckpt_dir, f"{dataset}_checkpoint.pth")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    class Wrapper(torch.nn.Module):
        def forward(self, x):
            enc = model.embedding(x)
            enc, _, _, _ = model.encoder(enc)
            return model.projection(enc)

    dummy = torch.randn(1, cfg["win_size"], cfg["enc_in"])
    ep = torch.export.export(Wrapper().eval(), (dummy,)).module()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset}_fp32.tflite")
    litert_torch.convert(ep, (dummy,)).export(path)
    print(f"[fp32]  {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="SMD", choices=list(PRESETS.keys()))
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--out_dir", default="exports")
    a = ap.parse_args()
    export(a.dataset, PRESETS[a.dataset], a.ckpt_dir, a.out_dir)


if __name__ == "__main__":
    main()
