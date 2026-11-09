"""Training dynamics: Test-II CER and training loss per epoch, for a readable
subset of representative configurations.

Replaces the earlier all-runs plot, which put thirteen curves on one axis with a
repeating ten-colour cycle, so that several pairs of runs were drawn in the same
colour and the raw checkpoint directory names appeared in the legend.

Left panel supports the schedule argument of the ablation section: the
convolutional-recurrent runs are flat well before epoch 25, while the transformer
and diffusion recognizers are still improving at epoch 80.  Right panel supports
the encoder diagnosis: the ConvNeXt run reaches a *lower* training loss than the
best model while its validation error stays more than twice as high.

Reads ``checkpoints/<run>/metrics.jsonl``; writes
``figures/training_curves/training_dynamics.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (display label, run dir, colour, linestyle)
RUNS = [
    ("CRNN-VGG, asp. 48$\\times$512", "abl_asp_48x512_s2", "#1f77b4", "-"),
    ("CRNN-VGG, 48$\\times$368, no aug.", "abl_aug_none_s1", "#17becf", "-"),
    ("SVTR-Small", "svtr_small_s1", "#2ca02c", "-"),
    ("DiffOCR", "diff_ocr_s1", "#9467bd", "-"),
    ("CRNN-ConvNeXt", "crnn_convnext_s1", "#d62728", "-"),
]

PRIOR_T2 = 28.65


def load(run: str) -> list[dict]:
    p = Path("checkpoints") / run / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("figures/training_curves"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.7))

    for label, run, colour, ls in RUNS:
        rows = load(run)
        # Test-II CER is logged on a sparse grid for the diffusion run.
        xs = [r["epoch"] + 1 for r in rows if "test_II_cer" in r]
        ys = [r["test_II_cer"] for r in rows if "test_II_cer" in r]
        if xs:
            ax1.plot(xs, ys, color=colour, linestyle=ls, linewidth=1.6, label=label,
                     marker="o" if len(xs) < 25 else None, markersize=3)
        ax2.plot([r["epoch"] + 1 for r in rows], [r["train_loss"] for r in rows],
                 color=colour, linestyle=ls, linewidth=1.6, label=label)

    ax1.axhline(PRIOR_T2, color="0.3", linestyle="--", linewidth=1.0)
    ax1.text(81, PRIOR_T2 + 0.8, "prior best (28.65)", fontsize=7.5, color="0.3", ha="right")
    ax1.axvline(25, color="0.55", linestyle=":", linewidth=1.0)
    ax1.text(26, 52, "25 epochs", fontsize=7.5, color="0.4")
    ax1.set_xlabel("epoch", fontsize=9)
    ax1.set_ylabel("Test-II CER (%)", fontsize=9)
    ax1.set_ylim(15, 60)
    ax1.set_xlim(0, 82)
    ax1.grid(alpha=0.3, linewidth=0.6)
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper right")

    ax2.set_yscale("log")
    ax2.set_xlabel("epoch", fontsize=9)
    ax2.set_ylabel("training loss (log scale)", fontsize=9)
    ax2.set_xlim(0, 82)
    ax2.grid(alpha=0.3, linewidth=0.6, which="both")
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        target = args.out / f"training_dynamics.{ext}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print("wrote", target)
    plt.close(fig)

    print("\nfinal-epoch summary")
    for label, run, _c, _l in RUNS:
        rows = load(run)
        f = rows[-1]
        print(f"  {label:34s} loss={f['train_loss']:.4f}  val CER={f['val_cer']:.2f}")


if __name__ == "__main__":
    main()
