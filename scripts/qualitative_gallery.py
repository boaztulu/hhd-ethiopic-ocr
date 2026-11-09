"""Render a qualitative-examples grid from a predictions CSV.

Reads ``results/<run>/predictions_<split>.csv`` and shows N random images with
their ground truth and prediction underneath, highlighting matches/mismatches.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import editdistance
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.dataset import SPLIT_PATHS


# Try to find an Ethiopic-capable font on the system
def _find_eth_font() -> str | None:
    for name in ["NotoSansEthiopic", "AbyssinicaSIL", "Kefa", "Ethiopic"]:
        match = next((f for f in fm.fontManager.ttflist if name.lower() in f.name.lower()), None)
        if match:
            return match.fname
    # fallback: any font claiming to support Ethiopic
    for f in fm.fontManager.ttflist:
        if "ethiopic" in f.fname.lower():
            return f.fname
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--n_samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    with open(args.predictions, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        print("no predictions")
        return

    split = args.predictions.stem.replace("predictions_", "")
    img_dir = SPLIT_PATHS[split]["img_dir"]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(rows), size=min(args.n_samples, len(rows)), replace=False)

    eth_font = _find_eth_font()
    font_kw = {"family": "Noto Sans Ethiopic"} if eth_font else {}

    cols = 3
    rowsN = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rowsN, cols, figsize=(cols * 6, rowsN * 2.5), constrained_layout=True)
    axes = np.array(axes).reshape(rowsN, cols)
    for ax in axes.flat:
        ax.axis("off")

    for k, i in enumerate(idx):
        ax = axes[k // cols, k % cols]
        r = rows[i]
        fn = r["filename"]
        gt = r["ground_truth"]
        pr = r["prediction"]
        ed = editdistance.eval(gt, pr)
        color = "green" if ed == 0 else "darkred"
        img_path = img_dir / fn
        if img_path.exists():
            im = Image.open(img_path).convert("RGB")
            ax.imshow(im)
        title = f"GT: {gt}\nPred: {pr}  (ED={ed})"
        ax.set_title(title, fontsize=9, color=color, loc="left", **font_kw)

    out = args.out or (Path("figures") / "qualitative" / f"{args.predictions.parent.name}_{split}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
