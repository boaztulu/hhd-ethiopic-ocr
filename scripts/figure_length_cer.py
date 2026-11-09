"""CER as a function of ground-truth line length, per model and split.

Motivation: the ImageNet-pretrained ConvNeXt encoder in ``src/models/crnn.py``
takes timm's stage-3 features (total stride 32), so a 48x512 input yields only
16 CTC timesteps.  CTC cannot emit a label longer than the number of timesteps,
so every line longer than 16 characters is unrepresentable.  Test-II lines are
systematically longer than Test-I lines, which is why the OOD damage is worse.

This figure makes that mechanism visible: CER against ground-truth length, with
the VGG backbone (T=127 at the same input) as a control.

Reads ``results/<run>/predictions_<split>.csv`` and writes
``figures/length_cer/length_cer.{pdf,png}`` plus the underlying numbers to
``tables/length_cer.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import editdistance
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, run directory, CTC timesteps at that run's input geometry)
MODELS = [
    ("CRNN-VGG (T=127)", "abl_asp_48x512_s2", 127),
    ("CRNN-ConvNeXt (T=16)", "crnn_convnext_s1", 16),
]

SPLITS = [("test_I", "Test-I (in-distribution)"), ("test_II", "Test-II (18th c.\\ OOD)")]

# Right-open bins on ground-truth character length.
BINS = [(1, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 32), (33, 100)]


def bin_label(lo: int, hi: int) -> str:
    return f"{lo}–{hi}" if hi < 100 else f"{lo}+"


def load(run: str, split: str) -> list[tuple[str, str]]:
    path = Path("results") / run / f"predictions_{split}.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["ground_truth"], r["prediction"]))
    return rows


def cer(rows: list[tuple[str, str]]) -> float:
    """Corpus CER = total edit distance / total reference characters, in percent.

    This matches the definition used by Belay et al. and by ``src/utils/metrics.py``;
    it is *not* the mean of per-line CERs.
    """
    num = sum(editdistance.eval(g, p) for g, p in rows)
    den = sum(len(g) for g, _ in rows)
    return 100.0 * num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("figures/length_cer"), type=Path)
    ap.add_argument("--csv", default=Path("tables/length_cer.csv"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    records = []
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True)

    for ax, (split, split_title) in zip(axes, SPLITS):
        for (label, run, _t), colour, marker in zip(
            MODELS, ["#1f77b4", "#d62728"], ["o", "s"]
        ):
            rows = load(run, split)
            xs, ys = [], []
            for lo, hi in BINS:
                sub = [r for r in rows if lo <= len(r[0]) <= hi]
                if not sub:
                    continue
                value = cer(sub)
                xs.append(bin_label(lo, hi))
                ys.append(value)
                records.append(
                    {
                        "run": run,
                        "model": label,
                        "split": split,
                        "len_bin": bin_label(lo, hi),
                        "n_lines": len(sub),
                        "cer": round(value, 4),
                    }
                )
            ax.plot(xs, ys, marker=marker, color=colour, label=label, linewidth=1.8, markersize=5)

        # Mark where the ConvNeXt CTC output budget runs out.
        ax.axvline(2.5, color="0.35", linestyle=":", linewidth=1.2)
        ax.text(
            2.58,
            ax.get_ylim()[1] * 0.02 + 78,
            "16-timestep\nCTC budget",
            fontsize=7.5,
            color="0.3",
            va="top",
        )
        ax.set_title(split_title.replace("\\ ", " "), fontsize=10)
        ax.set_xlabel("ground-truth line length (characters)", fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("CER (\\%)".replace("\\", ""), fontsize=9)
    axes[0].set_ylim(0, 85)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        target = args.out / f"length_cer.{ext}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print("wrote", target)
    plt.close(fig)

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "model", "split", "len_bin", "n_lines", "cer"])
        w.writeheader()
        w.writerows(records)
    print("wrote", args.csv)

    # Console summary: corpus CER for lines of at most 16 and of more than 16 characters.
    print("\nlen<=16 vs len>16 corpus CER")
    for label, run, _t in MODELS:
        for split, _title in SPLITS:
            rows = load(run, split)
            short = [r for r in rows if len(r[0]) <= 16]
            long_ = [r for r in rows if len(r[0]) > 16]
            print(
                f"  {label:24s} {split:8s} all={cer(rows):6.2f}  "
                f"<=16={cer(short):6.2f} (n={len(short)})  >16={cer(long_):6.2f} (n={len(long_)})"
            )


if __name__ == "__main__":
    main()
