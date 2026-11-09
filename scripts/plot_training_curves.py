"""Plot training curves from one or more ``metrics.jsonl`` files.

Usage:
    python scripts/plot_training_curves.py checkpoints/*/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with open(p) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figures") / "training_curves")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for p in args.paths:
        runs[p.parent.name] = load_jsonl(p)

    # Plot 1: train loss + val CER per run
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name, rows in runs.items():
        if not rows:
            continue
        ep = [r["epoch"] for r in rows]
        tl = [r.get("train_loss") for r in rows]
        vc = [r.get("val_cer") for r in rows]
        axes[0].plot(ep, tl, label=name, marker="o", markersize=3)
        axes[1].plot(ep, vc, label=name, marker="o", markersize=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss"); axes[0].set_title("Training loss")
    axes[0].set_yscale("log"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val CER (%)"); axes[1].set_title("Validation CER")
    axes[1].axhline(16.41, ls="--", color="black", alpha=0.5, label="HPopt-Attn-CTC T1 (16.41)")
    axes[1].axhline(28.65, ls=":",  color="black", alpha=0.5, label="HPopt-Attn-CTC T2 (28.65)")
    axes[1].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"loss_and_val_cer.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {args.out/'loss_and_val_cer.png'}")

    # Plot 2: per-epoch Test-I and Test-II CER (when reported)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name, rows in runs.items():
        ep_t1 = [r["epoch"] for r in rows if r.get("test_I_cer") is not None]
        cer_t1 = [r["test_I_cer"] for r in rows if r.get("test_I_cer") is not None]
        ep_t2 = [r["epoch"] for r in rows if r.get("test_II_cer") is not None]
        cer_t2 = [r["test_II_cer"] for r in rows if r.get("test_II_cer") is not None]
        if ep_t1:
            axes[0].plot(ep_t1, cer_t1, label=name, marker="o", markersize=4)
        if ep_t2:
            axes[1].plot(ep_t2, cer_t2, label=name, marker="o", markersize=4)
    axes[0].axhline(16.41, ls="--", color="black", alpha=0.5)
    axes[0].set_title("Test-I CER (IID)  · dashed = HPopt-Attn-CTC 16.41")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("CER (%)"); axes[0].legend(fontsize=8)
    axes[1].axhline(28.65, ls="--", color="black", alpha=0.5)
    axes[1].set_title("Test-II CER (OOD)  · dashed = HPopt-Attn-CTC 28.65")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("CER (%)"); axes[1].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"test_cer.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {args.out/'test_cer.png'}")


if __name__ == "__main__":
    main()
