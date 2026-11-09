"""Corruption-robustness curves, one panel per corruption.

The sweep in ``results/<run>/robustness_<split>.csv`` varies three corruptions
whose level parameters are not commensurable: gaussian-blur kernel size runs 0-9,
noise variance 0-300 and brightness reduction 0-0.5.  Plotting them against a
single shared x-axis, as an earlier version of this figure did, compresses blur
and brightness into a vertical line at the origin and hides the only interesting
comparison.  Each corruption therefore gets its own panel and its own axis.

Writes ``figures/robustness/robustness_<split>.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("blur", "gaussian blur (kernel size)", "#1f77b4", "o"),
    ("noise", "additive gaussian noise (variance)", "#ff7f0e", "s"),
    ("brightness", "brightness reduction (fraction)", "#2ca02c", "^"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="crnn_vgg_s1")
    ap.add_argument("--split", default="test_II")
    ap.add_argument("--out", default=Path("figures/robustness"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    src = Path("results") / args.run / f"robustness_{args.split}.csv"
    series = defaultdict(list)
    with open(src, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            series[r["corruption"]].append((float(r["level"]), float(r["cer"])))
    for k in series:
        series[k].sort()

    clean = series["blur"][0][1]  # level 0 of the blur sweep is the uncorrupted image

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, (key, xlabel, colour, marker) in zip(axes, PANELS):
        xs = [x for x, _ in series[key]]
        ys = [y for _, y in series[key]]
        ax.plot(xs, ys, marker=marker, color=colour, linewidth=1.8, markersize=5)
        ax.axhline(clean, color="0.45", linestyle="--", linewidth=1.0)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Test-II CER (%)", fontsize=9)
    # Label the reference line in the noise panel, which is the one with free space.
    axes[1].text(150, clean + 3.0, f"uncorrupted ({clean:.1f})", fontsize=7.5,
                 color="0.35", ha="center")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        target = args.out / f"robustness_{args.split}.{ext}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print("wrote", target)
    plt.close(fig)

    for key, _lab, _c, _m in PANELS:
        pts = ", ".join(f"{x:g}:{y:.2f}" for x, y in series[key])
        print(f"  {key:11s} {pts}")


if __name__ == "__main__":
    main()
