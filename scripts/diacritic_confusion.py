"""Character-level confusion analysis on a predictions CSV.

Reads ``results/<run>/predictions_<split>.csv`` (columns: filename, ground_truth, prediction),
computes a substitution-confusion matrix using a Wagner-Fischer alignment, and emits:

  * tables/confusion_<run>_<split>_top<K>.csv: top-K substituted pairs by count
  * figures/confusion_<run>_<split>_top<K>.{png,pdf}
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def align(a: str, b: str) -> list[tuple[str, str, str]]:
    """Return list of (op, a_char, b_char), op in {'=', 'sub', 'ins', 'del'}.

    ins = char inserted into prediction (i.e. b has char that a doesn't),
    del = char deleted from prediction (a has char that b doesn't).
    """
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i, j] == dp[i - 1, j - 1]:
            ops.append(("=", a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            ops.append(("ins", "", b[j - 1])); j -= 1
        else:
            ops.append(("del", a[i - 1], "")); i -= 1
    ops.reverse()
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path,
                    help="results/<run>/predictions_<split>.csv")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    pred_path = args.predictions
    run = pred_path.parent.name
    split = pred_path.stem.replace("predictions_", "")
    out_dir = args.out_dir or pred_path.parent
    fig_dir = Path("figures") / "confusion"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_counter: Counter[tuple[str, str]] = Counter()
    ins_counter: Counter[str] = Counter()
    del_counter: Counter[str] = Counter()
    n_pairs = 0
    n_errors = 0
    with open(pred_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_pairs += 1
            ops = align(row["ground_truth"], row["prediction"])
            for op, a, b in ops:
                if op == "sub":
                    sub_counter[(a, b)] += 1; n_errors += 1
                elif op == "ins":
                    ins_counter[b] += 1; n_errors += 1
                elif op == "del":
                    del_counter[a] += 1; n_errors += 1

    top_sub = sub_counter.most_common(args.top_k)
    top_ins = ins_counter.most_common(args.top_k)
    top_del = del_counter.most_common(args.top_k)

    # CSV outputs
    csv_path = Path("tables") / f"confusion_{run}_{split}_top{args.top_k}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "from_char", "to_char", "count"])
        for (a, b), c in top_sub:
            w.writerow(["sub", a, b, c])
        for ch, c in top_ins:
            w.writerow(["ins", "", ch, c])
        for ch, c in top_del:
            w.writerow(["del", ch, "", c])
    print(f"wrote {csv_path}  total_errors={n_errors}  pairs={n_pairs}")

    # Bar chart of top substitutions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    if top_sub:
        labels = [f"{a}→{b}" for (a, b), _ in top_sub]
        counts = [c for _, c in top_sub]
        axes[0].barh(labels, counts, color="indianred")
        axes[0].invert_yaxis()
        axes[0].set_title(f"Top {len(top_sub)} substitutions")
        axes[0].set_xlabel("count")
    if top_ins:
        labels = [c for c, _ in top_ins]
        counts = [n for _, n in top_ins]
        axes[1].barh(labels, counts, color="steelblue")
        axes[1].invert_yaxis()
        axes[1].set_title(f"Top {len(top_ins)} inserted chars")
        axes[1].set_xlabel("count")
    if top_del:
        labels = [c for c, _ in top_del]
        counts = [n for _, n in top_del]
        axes[2].barh(labels, counts, color="darkorange")
        axes[2].invert_yaxis()
        axes[2].set_title(f"Top {len(top_del)} deleted chars")
        axes[2].set_xlabel("count")
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"confusion_{run}_{split}_top{args.top_k}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote figures/confusion/confusion_{run}_{split}_top{args.top_k}.png")


if __name__ == "__main__":
    main()
