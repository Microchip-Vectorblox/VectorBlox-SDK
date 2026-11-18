"""anomaly_train.py — synthetic data loader and training for AnomalyTransformer.

Usage:
    python anomaly_train.py                          # train all four datasets
    python anomaly_train.py --dataset SMD            # SMD only (~60s CPU)
    python anomaly_train.py --dataset SMD --mode test
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.AnomalyTransformer import AnomalyTransformer


# ── synthetic data ────────────────────────────────────────────────────────────


class SyntheticSegLoader(Dataset):
    """Random-walk + sinusoidal multivariate time series.

    Train windows are clean; val/test windows contain injected spike anomalies.
    """

    def __init__(
        self,
        n_windows=2000,
        win_size=100,
        n_channels=38,
        anomaly_ratio=0.05,
        mode="train",
        seed=42,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)

        total_len = n_windows + win_size
        t = np.linspace(0, 4 * np.pi, total_len)
        freqs = rng.uniform(0.5, 3.0, size=(n_channels,))
        phases = rng.uniform(0, 2 * np.pi, size=(n_channels,))

        signal = np.cumsum(rng.normal(0, 0.1, size=(total_len, n_channels)), axis=0)
        signal += np.stack(
            [np.sin(freqs[c] * t + phases[c]) for c in range(n_channels)], axis=1
        )
        signal = (signal - signal.mean(0)) / (signal.std(0) + 1e-8)
        signal = signal.astype(np.float32)

        windows = np.stack([signal[i : i + win_size] for i in range(n_windows)])
        labels = np.zeros((n_windows, win_size), dtype=np.float32)

        if mode != "train":
            n_anom = max(1, int(n_windows * anomaly_ratio))
            for idx in rng.choice(n_windows, size=n_anom, replace=False):
                start = rng.integers(0, win_size - 10)
                end = min(start + rng.integers(5, 20), win_size)
                ch = rng.integers(0, n_channels)
                windows[idx, start:end, ch] += rng.uniform(3, 6)
                labels[idx, start:end] = 1.0

        self.windows = windows
        self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return self.windows[i], self.labels[i]


def get_synthetic_loader(
    batch_size=256,
    win_size=100,
    n_channels=38,
    n_windows=2000,
    anomaly_ratio=0.05,
    mode="train",
    seed=42,
):
    ds = SyntheticSegLoader(
        n_windows=n_windows,
        win_size=win_size,
        n_channels=n_channels,
        anomaly_ratio=anomaly_ratio,
        mode=mode,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=0,
        drop_last=False,
    )


# ── dataset presets ───────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "SMD": dict(input_c=38, output_c=38, win_size=100, anormly_ratio=0.5),
    "MSL": dict(input_c=55, output_c=55, win_size=100, anormly_ratio=1.0),
    "SMAP": dict(input_c=25, output_c=25, win_size=100, anormly_ratio=1.0),
    "PSM": dict(input_c=25, output_c=25, win_size=100, anormly_ratio=1.0),
}


# ── association loss ──────────────────────────────────────────────────────────


def my_kl_loss(p, q):
    return torch.mean(
        torch.sum(p * (torch.log(p + 1e-4) - torch.log(q + 1e-4)), dim=-1), dim=1
    )


def association_losses(series, prior, win_size, k):
    s_loss = p_loss = 0.0
    for u in range(len(prior)):
        norm = prior[u] / torch.unsqueeze(prior[u].sum(-1), -1).repeat(
            1, 1, 1, win_size
        )
        s_loss += torch.mean(my_kl_loss(series[u], norm.detach())) + torch.mean(
            my_kl_loss(norm.detach(), series[u])
        )
        p_loss += torch.mean(my_kl_loss(norm, series[u].detach())) + torch.mean(
            my_kl_loss(series[u].detach(), norm)
        )
    return s_loss / len(prior), p_loss / len(prior)


# ── train ─────────────────────────────────────────────────────────────────────


def train_one(dataset_name, cfg, args, device):
    print(
        f"\n{'=' * 60}\n  TRAIN  {dataset_name}  "
        f"(channels={cfg['input_c']}, win={cfg['win_size']})\n{'=' * 60}"
    )

    train_loader = get_synthetic_loader(
        batch_size=args.batch_size,
        win_size=cfg["win_size"],
        n_channels=cfg["input_c"],
        n_windows=args.n_windows,
        mode="train",
        seed=0,
    )
    val_loader = get_synthetic_loader(
        batch_size=args.batch_size,
        win_size=cfg["win_size"],
        n_channels=cfg["input_c"],
        n_windows=args.n_windows // 5,
        anomaly_ratio=cfg["anormly_ratio"] / 100,
        mode="val",
        seed=1,
    )

    model = AnomalyTransformer(
        win_size=cfg["win_size"],
        enc_in=cfg["input_c"],
        c_out=cfg["output_c"],
        d_model=args.d_model,
        n_heads=8,
        e_layers=3,
        d_ff=args.d_model,
        dropout=0.0,
        activation="gelu",
        output_attention=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            out, series, prior, _ = model(x)
            rec = criterion(out, x)
            sl, pl = association_losses(series, prior, cfg["win_size"], args.k)
            (rec - args.k * sl).backward(retain_graph=True)
            (rec + args.k * pl).backward()
            optimizer.step()
            losses.append(rec.item())
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * (0.5**epoch)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                out, series, prior, _ = model(x)
                rec = criterion(out, x)
                sl, _ = association_losses(series, prior, cfg["win_size"], args.k)
                val_losses.append((rec - args.k * sl).item())

        val_loss = np.mean(val_losses)
        print(
            f"  Epoch {epoch:3d}/{args.num_epochs}  "
            f"train={np.mean(losses):.6f}  val={val_loss:.6f}  "
            f"({time.time() - t0:.1f}s)"
        )

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(args.model_save_path, exist_ok=True)
            ckpt = os.path.join(args.model_save_path, f"{dataset_name}_checkpoint.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"    → {ckpt}")


# ── test ──────────────────────────────────────────────────────────────────────


def test_one(dataset_name, cfg, args, device):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    print(f"\n{'=' * 60}\n  TEST   {dataset_name}\n{'=' * 60}")
    ckpt = os.path.join(args.model_save_path, f"{dataset_name}_checkpoint.pth")
    if not os.path.exists(ckpt):
        print(f"  No checkpoint at {ckpt}. Run --mode train first.")
        return

    model = AnomalyTransformer(
        win_size=cfg["win_size"],
        enc_in=cfg["input_c"],
        c_out=cfg["output_c"],
        d_model=args.d_model,
        n_heads=8,
        e_layers=3,
        d_ff=args.d_model,
        dropout=0.0,
        activation="gelu",
        output_attention=True,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    crit = nn.MSELoss(reduction="none")
    T = 50

    def collect(loader):
        energies, labels = [], []
        with torch.no_grad():
            for x, lbl in loader:
                x = x.to(device)
                out, series, prior, _ = model(x)
                rec = torch.mean(crit(x, out), dim=-1)
                sl = pl = None
                for u in range(len(prior)):
                    norm = prior[u] / torch.unsqueeze(prior[u].sum(-1), -1).repeat(
                        1, 1, 1, cfg["win_size"]
                    )
                    kl = (
                        my_kl_loss(series[u], norm.detach())
                        + my_kl_loss(norm, series[u].detach())
                    ) * T
                    sl = kl if sl is None else sl + kl
                score = (torch.softmax(-sl, dim=-1) * rec).cpu().numpy()
                energies.append(score)
                labels.append(lbl.numpy())
        return np.concatenate(energies).ravel(), np.concatenate(labels).ravel()

    train_e, _ = collect(
        get_synthetic_loader(
            args.batch_size,
            cfg["win_size"],
            cfg["input_c"],
            args.n_windows,
            mode="train",
            seed=0,
        )
    )
    test_e, gt = collect(
        get_synthetic_loader(
            args.batch_size,
            cfg["win_size"],
            cfg["input_c"],
            args.n_windows // 5,
            anomaly_ratio=cfg["anormly_ratio"] / 100,
            mode="thre",
            seed=2,
        )
    )
    thresh = np.percentile(
        np.concatenate([train_e, test_e]), 100 - cfg["anormly_ratio"]
    )
    pred = (test_e > thresh).astype(int)
    acc = accuracy_score(gt.astype(int), pred)
    p, r, f1, _ = precision_recall_fscore_support(
        gt.astype(int), pred, average="binary", zero_division=0
    )
    print(f"  Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dataset", default="all", choices=["all", "SMD", "MSL", "SMAP", "PSM"]
    )
    ap.add_argument("--mode", default="train", choices=["train", "test", "both"])
    ap.add_argument("--num_epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_windows", type=int, default=2000)
    ap.add_argument("--model_save_path", default="checkpoints")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    datasets = list(DATASET_CONFIGS.keys()) if a.dataset == "all" else [a.dataset]
    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        if a.mode in ("train", "both"):
            train_one(ds, cfg, a, device)
        if a.mode in ("test", "both"):
            test_one(ds, cfg, a, device)


if __name__ == "__main__":
    main()
