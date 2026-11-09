"""Grouped bar chart: Test-I and Test-II CER, published baselines against ours.

Reads ``tables/main_results.csv`` (produced by ``build_main_table.py``) and emits
``figures/comparison_bars/comparison_bars.{pdf,png}``.

Only converged configurations are plotted, and each run directory is mapped to a
readable model name; the run-name labels the earlier version of this script used
are not suitable for publication.  The PARSeq run is excluded because it stopped
after 5 of 80 epochs and is not a converged result.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Published rows, in the order they should appear.
PUBLISHED = [
    ("Plain-CTC", "Plain-CTC (Belay 2020a)"),
    ("Attn-CTC", "Attn-CTC (Belay 2021a)"),
    ("HPopt-Plain-CTC", "HPopt-Plain-CTC"),
    ("HPopt-Attn-CTC", "HPopt-Attn-CTC (prior best)"),
    ("CRNN", "CRNN (Shi 2016)"),
    ("ASTER", "ASTER (Shi 2018)"),
    ("SVTR", "SVTR (Du 2022)"),
    ("ABINet", "ABINet (Fang 2021)"),
    ("Best human", "Best human"),
    ("Avg human", "Avg human"),
]

# Our rows: display name -> run key in main_results.csv.
# Runs of one configuration are averaged and shown once.
OURS = [
    ("CRNN-VGG, asp. 48$\\times$512", ["abl_asp_48x512_s1", "abl_asp_48x512_s2", "abl_asp_48x512_s3"]),
    ("CRNN-VGG, 48$\\times$368", ["crnn_vgg_s1", "crnn_vgg_s2", "crnn_vgg_s3"]),
    ("CRNN-VGG, heavy aug.", ["abl_aug_heavy_s1"]),
    ("CRNN-VGG, no aug.", ["abl_aug_none_s1", "abl_pub_48x368_s1"]),
    ("SVTR-Small", ["svtr_small_s1"]),
    ("DiffOCR", ["diff_ocr_s1"]),
    ("CRNN-ConvNeXt", ["crnn_convnext_s1"]),
]

PRIOR_T1, PRIOR_T2 = 16.41, 28.65


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=Path("tables/main_results.csv"), type=Path)
    ap.add_argument("--out", default=Path("figures/comparison_bars"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    by_method = {}
    for r in csv.DictReader(open(args.csv, encoding="utf-8")):
        try:
            by_method[r["method"]] = (float(r["T1_CER"]), float(r["T2_CER"]))
        except ValueError:
            continue

    labels, t1, t2, is_ours = [], [], [], []

    for key, display in PUBLISHED:
        hit = next((m for m in by_method if m.startswith(key)), None)
        if hit is None:
            print(f"  [missing published row: {key}]")
            continue
        labels.append(display)
        t1.append(by_method[hit][0])
        t2.append(by_method[hit][1])
        is_ours.append(False)

    for display, runs in OURS:
        vals = [by_method[r] for r in runs if r in by_method]
        if not vals:
            print(f"  [missing run(s): {runs}]")
            continue
        labels.append(display + (f" ({len(vals)} runs)" if len(vals) > 1 else ""))
        t1.append(float(np.mean([v[0] for v in vals])))
        t2.append(float(np.mean([v[1] for v in vals])))
        is_ours.append(True)

    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11.5, 4.2))

    for i in range(len(labels)):
        base = "#c0392b" if is_ours[i] else "#2c6fad"
        ax.bar(x[i] - width / 2, t1[i], width, color=base, edgecolor="black", linewidth=0.4)
        ax.bar(x[i] + width / 2, t2[i], width, color=base, edgecolor="black",
               linewidth=0.4, alpha=0.55, hatch="///")

    ax.axhline(PRIOR_T1, color="0.25", linestyle="--", linewidth=1.0)
    ax.axhline(PRIOR_T2, color="0.25", linestyle=":", linewidth=1.0)
    ax.text(-0.55, PRIOR_T1 + 0.5, "prior best, Test-I", fontsize=7.5,
            color="0.25", ha="left")
    ax.text(-0.55, PRIOR_T2 + 0.5, "prior best, Test-II", fontsize=7.5,
            color="0.25", ha="left")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#2c6fad", edgecolor="black", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, facecolor="#c0392b", edgecolor="black", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.55", edgecolor="black", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.55", edgecolor="black", linewidth=0.4,
                      alpha=0.55, hatch="///"),
    ]
    ax.legend(handles, ["Belay et al. (published)", "this study", "Test-I (IID)",
                        "Test-II (18th c. OOD)"],
              fontsize=8, frameon=False, ncol=2, loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("CER (%)", fontsize=9)
    ax.set_ylim(0, 48)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        target = args.out / f"comparison_bars.{ext}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print("wrote", target)
    plt.close(fig)


if __name__ == "__main__":
    main()
