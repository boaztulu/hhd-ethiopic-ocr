"""Redraw the summary figures at print size (12.2 cm wide, 6.5-8 pt text).

Data sources are the same as the original plotting scripts in the repository
(scripts/figure_*.py, scripts/regen_ethiopic_figs.py); only geometry, type size
and the colour mapping change. Colour follows the entity across all figures.
"""
import csv, io, json, sys, zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # project root
HERE = Path(__file__).parent
OUT = ROOT / "results" / "analysis"
FIG = ROOT / "figures" / "analysis"
ZIPS = ROOT / "HHD-Ethiopic-data"
W = 4.80  # figure width in inches (12.2 cm)

# entity -> colour (validated categorical slots 1-5; published results in neutral grey)
C = {"vgg_asp": "#2a78d6", "convnext": "#eb6834", "svtr": "#1baf7a", "diffocr": "#eda100",
     "vgg_368": "#e87ba4", "published": "#8a8985", "ink": "#0b0b0b", "muted": "#52514e",
     "train": "#b9b8b3", "test_I": "#6250d6", "test_II": "#e34948"}

fm.fontManager.addfont(str(ROOT / "cache/NotoSansEthiopic.ttf"))
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Noto Sans Ethiopic"], "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.1, "lines.markersize": 3.2,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.color": "#e4e3df", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)
    print("wrote", name, flush=True)


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return [(r[0].strip(), r[1].rstrip(" ").rstrip()) for r in csv.reader(f) if len(r) >= 2]


def preds(run, split):
    with open(ROOT / f"results/{run}/predictions_{split}.csv", encoding="utf-8") as f:
        return [(r["ground_truth"], r["prediction"]) for r in csv.DictReader(f)]


# ---------------------------------------------------------------- samples
def fig_samples():
    D = ROOT / "HHD-Ethiopic-data"
    splits = [("Train", D / "train/train_raw/image_text_pairs_train.csv", "train_raw.zip", "train_raw/image_train/"),
              ("Test-I", D / "test/test_rand/image_text_pairs_test_rand.csv", "test.zip", "test/test_rand/image_rand/"),
              ("Test-II", D / "test/test_18th/image_text_pairs_test_18th.csv", "test.zip", "test/test_18th/image_18th/")]
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(3, 2, figsize=(W, 1.45), gridspec_kw={"hspace": 0.55, "wspace": 0.06})
    for r, (label, csvp, z, prefix) in enumerate(splits):
        rows = [x for x in read_csv(csvp) if 8 <= len(x[1]) <= 22]
        pick = rng.choice(len(rows), size=3, replace=False)[:2]
        with zipfile.ZipFile(ZIPS / z) as zf:
            for c, i in enumerate(pick):
                fn, t = rows[i]
                ax = axes[r, c]
                ax.imshow(Image.open(io.BytesIO(zf.read(prefix + fn))).convert("L"), cmap="gray", aspect="equal")
                ax.set_title(t, fontsize=7.5, pad=2)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
        axes[r, 0].text(-0.03, 0.5, label, transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=7, fontweight="bold")
    save(fig, "eda_samples")


# ---------------------------------------------------------------- geometry
def fig_geometry():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.55), gridspec_kw={"wspace": 0.28})
    spec = [("train", "Train", dict(histtype="stepfilled", color=C["train"], alpha=0.75)),
            ("test_I", "Test-I", dict(histtype="step", color=C["test_I"], linestyle="--", linewidth=1.0)),
            ("test_II", "Test-II", dict(histtype="step", color=C["test_II"], linewidth=1.2))]
    for key, label, kw in spec:
        g = np.load(OUT / f"geom_{key}.npz")
        a1.hist(g["W"], bins=np.arange(0, 1401, 25), density=True, label=label, **kw)
        a2.hist(g["W"] / g["H"], bins=np.arange(0, 30.5, 0.5), density=True, label=label, **kw)
    a1.set_xlabel("line width (pixels)"); a1.set_ylabel("density"); a1.set_xlim(0, 1400)
    a2.set_xlabel("aspect ratio (width / height)"); a2.set_xlim(0, 30)
    for x, txt, side in ((368 / 48, "368:48", "right"), (512 / 48, "512:48", "left")):
        a2.axvline(x, color=C["ink"], linestyle=":", linewidth=0.8)
        a2.text(x + (0.4 if side == "left" else -0.4), a2.get_ylim()[1] * 0.97, txt, fontsize=6,
                color=C["muted"], va="top", ha=side)
    a1.legend(loc="upper right", handlelength=1.6)
    for a in (a1, a2):
        a.set_yticks([])
    save(fig, "image_dimensions")


# ---------------------------------------------------------------- character frequency
def fig_charfreq(k=40):
    rows = list(csv.DictReader(open(ROOT / "tables/char_frequency_top50.csv", encoding="utf-8")))[:k]
    fig, ax = plt.subplots(figsize=(W, 1.35))
    ax.bar(range(k), [int(r["train"]) for r in rows], color=C["muted"], width=0.72)
    ax.set_yscale("log"); ax.set_xticks(range(k))
    ax.set_xticklabels([r["char"] for r in rows], fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_ylabel("training count"); ax.set_xlim(-0.7, k - 0.3); ax.grid(axis="x", visible=False)
    save(fig, "char_frequency")


# ---------------------------------------------------------------- comparison
def metric(run, split):
    return json.load(open(ROOT / f"results/{run}/metrics.json"))[split]["cer"]


def fig_comparison():
    pub = [("Plain-CTC", 20.88, 33.56), ("Attn-CTC", 19.42, 33.07), ("HPopt-Plain-CTC", 19.42, 32.01),
           ("HPopt-Attn-CTC", 16.41, 28.65), ("CRNN", 21.04, 29.86), ("ASTER", 24.43, 35.13),
           ("SVTR", 19.78, 30.82), ("ABINet", 21.49, 32.76), ("Best human", 25.39, 33.20)]
    ours = [("CRNN-VGG, 48×512 asp. (3 seeds)", ["abl_asp_48x512_s1", "abl_asp_48x512_s2", "abl_asp_48x512_s3"], "vgg_asp"),
            ("CRNN-VGG, 48×368 stretch (3 seeds)", ["crnn_vgg_s1", "crnn_vgg_s2_reeval", "crnn_vgg_s3"], "vgg_368"),
            ("DiffOCR", ["diff_ocr_s1"], "diffocr"), ("SVTR-Small", ["svtr_small_s1"], "svtr"),
            ("CRNN-ConvNeXt", ["crnn_convnext_s1"], "convnext")]
    rows = [(n, a, b, C["published"]) for n, a, b in pub]
    rows += [(n, np.mean([metric(r, "test_I") for r in rs]), np.mean([metric(r, "test_II") for r in rs]), C[e])
             for n, rs, e in ours]
    y = np.arange(len(rows))[::-1].astype(float)
    y[len(pub):] -= 0.8  # gap between the two groups
    fig, ax = plt.subplots(figsize=(W, 2.15))
    for yi, (n, t1, t2, col) in zip(y, rows):
        ax.plot([t1, t2], [yi, yi], color=col, linewidth=0.9, zorder=2)
        ax.plot(t1, yi, "o", color=col, markersize=3.6, zorder=3)
        ax.plot(t2, yi, "s", color=col, markersize=3.6, markerfacecolor="white", markeredgewidth=1.0, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    for x, txt in ((16.41, "prior best, Test-I"), (28.65, "prior best, Test-II")):
        ax.axvline(x, color=C["ink"], linestyle="--", linewidth=0.7, zorder=1)
        ax.text(x + 0.3, y[0] + 0.55, txt, fontsize=6, color=C["muted"], va="bottom")
    ax.set_xlim(10, 46); ax.set_ylim(y[-1] - 0.7, y[0] + 1.6); ax.set_xlabel("CER (%)")
    ax.grid(axis="y", visible=False)
    ax.text(45.6, (y[0] + y[len(pub) - 1]) / 2, "Belay et al.", rotation=90, ha="right", va="center", fontsize=6.5, color=C["muted"])
    ax.text(45.6, (y[len(pub)] + y[-1]) / 2, "this study", rotation=90, ha="right", va="center", fontsize=6.5, color=C["muted"])
    ax.axhline((y[len(pub) - 1] + y[len(pub)]) / 2, color="#cfcec9", linewidth=0.6)
    h = [plt.Line2D([], [], marker="o", color=C["muted"], linestyle="none", markersize=3.6, label="Test-I (IID)"),
         plt.Line2D([], [], marker="s", color=C["muted"], markerfacecolor="white", linestyle="none", markersize=3.6, label="Test-II (18th c.)")]
    # empty band right of the prior-best line, above the CRNN-ConvNeXt row, so no mark is covered
    ax.legend(handles=h, loc="center right", bbox_to_anchor=(44.2, (y[len(pub)] + y[-2]) / 2),
              bbox_transform=ax.transData)
    save(fig, "comparison_bars")


# ---------------------------------------------------------------- training dynamics
def fig_dynamics():
    runs = [("CRNN-VGG, 48×512 asp.", "abl_asp_48x512_s2", "vgg_asp", "-"),
            ("CRNN-VGG, 48×368, no aug.", "abl_aug_none_s1", "vgg_368", "--"),
            ("SVTR-Small", "svtr_small_s1", "svtr", "-"),
            ("DiffOCR", "diff_ocr_s1", "diffocr", "-"),
            ("CRNN-ConvNeXt", "crnn_convnext_s1", "convnext", "-.")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.85), gridspec_kw={"wspace": 0.36})
    for label, run, e, ls in runs:
        rows = [json.loads(l) for l in open(ROOT / f"checkpoints/{run}/metrics.jsonl") if l.strip()]
        xs = [r["epoch"] + 1 for r in rows if "test_II_cer" in r]
        ys = [r["test_II_cer"] for r in rows if "test_II_cer" in r]
        a1.plot(xs, ys, color=C[e], linestyle=ls, label=label, marker="o" if len(xs) < 25 else None, markersize=2.4)
        a2.plot([r["epoch"] + 1 for r in rows], [r["train_loss"] for r in rows], color=C[e], linestyle=ls)
    a1.axhline(28.65, color=C["ink"], linestyle=(0, (3, 2)), linewidth=0.7, label="prior best, 28.65")
    a1.axvline(25, color=C["muted"], linestyle=":", linewidth=0.7)
    a1.text(27.5, 57, "epoch 25", fontsize=6, color=C["muted"], va="top")  # clear of the DiffOCR curve
    a1.set_ylim(15, 60); a1.set_xlim(0, 82); a1.set_xlabel("epoch"); a1.set_ylabel("Test-II CER (%)")
    a2.set_yscale("log"); a2.set_xlim(0, 82); a2.set_xlabel("epoch"); a2.set_ylabel("training loss (log)")
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13),
               handlelength=2.2, columnspacing=1.2)
    save(fig, "training_dynamics")


# ---------------------------------------------------------------- length vs CER
def fig_length():
    import editdistance
    bins = [(1, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 32), (33, 100)]
    labels = [f"{a}–{b}" if b < 100 else f"{a}+" for a, b in bins]
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.65), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (split, title) in zip(axes, (("test_I", "Test-I"), ("test_II", "Test-II"))):
        for name, run, e, mk in (("CRNN-VGG (127 frames)", "abl_asp_48x512_s2", "vgg_asp", "o"),
                                 ("CRNN-ConvNeXt (16 frames)", "crnn_convnext_s1", "convnext", "s")):
            rows = preds(run, split)
            ys = []
            for lo, hi in bins:
                sub = [(g, p) for g, p in rows if lo <= len(g) <= hi]
                ys.append(100 * sum(editdistance.eval(g, p) for g, p in sub) / sum(len(g) for g, _ in sub))
            ax.plot(range(len(bins)), ys, marker=mk, color=C[e], label=name)
        ax.axvline(2.5, color=C["ink"], linestyle=":", linewidth=0.8)
        ax.text(2.4, 82, "16-frame\nlimit", fontsize=6, color=C["muted"], va="top", ha="right")
        ax.set_xticks(range(len(bins))); ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title(title, pad=2); ax.set_xlabel("ground-truth length (characters)"); ax.set_ylim(0, 85)
    axes[0].set_ylabel("CER (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))
    save(fig, "length_cer")


# ---------------------------------------------------------------- confusion profile
def align(a, b):
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32); dp[:, 0] = np.arange(m + 1); dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    ops, i, j = [], m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i, j] == dp[i - 1, j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            ops.append(("ins", "", b[j - 1])); j -= 1
        else:
            ops.append(("del", a[i - 1], "")); i -= 1
    return ops


def fig_confusion(k=20):
    sub, ins, dele = Counter(), Counter(), Counter()
    for g, p in preds("abl_asp_48x512_s2", "test_II"):
        for op, a, b in align(g, p):
            if op == "sub": sub[(a, b)] += 1
            elif op == "ins": ins[b] += 1
            else: dele[a] += 1
    name = {" ": "space"}
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.45), gridspec_kw={"wspace": 0.42})
    panels = [("substitutions", [(f"{a}→{b}", c) for (a, b), c in sub.most_common(k)]),
              ("insertions", [(name.get(a, a), c) for a, c in ins.most_common(k)]),
              ("deletions", [(name.get(a, a), c) for a, c in dele.most_common(k)])]
    for ax, (title, items) in zip(axes, panels):
        ax.barh(range(len(items)), [c for _, c in items], color=C["vgg_asp"], height=0.7)
        ax.set_yticks(range(len(items))); ax.set_yticklabels([s for s, _ in items], fontsize=7)
        ax.invert_yaxis(); ax.set_title(title); ax.set_xlabel("count"); ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, pad=1.5)
    save(fig, "confusion_test_II")
    return sub, ins, dele


# ---------------------------------------------------------------- robustness
def fig_robustness():
    series = {}
    for r in csv.DictReader(open(ROOT / "results/crnn_vgg_s1/robustness_test_II.csv", encoding="utf-8")):
        series.setdefault(r["corruption"], []).append((float(r["level"]), float(r["cer"])))
    clean = sorted(series["blur"])[0][1]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.45), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, (key, xl) in zip(axes, (("blur", "gaussian blur kernel (px)"), ("noise", "noise variance"),
                                    ("brightness", "brightness reduction"))):
        xs, ys = zip(*sorted(series[key]))
        ax.plot(xs, ys, marker="o", color=C["vgg_368"], markersize=3)
        ax.axhline(clean, color=C["ink"], linestyle="--", linewidth=0.7)
        ax.set_xlabel(xl)
    axes[0].set_ylabel("Test-II CER (%)")
    axes[1].text(150, clean + 4, f"uncorrupted ({clean:.1f})", fontsize=6, color=C["muted"], ha="center")
    save(fig, "robustness_test_II")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["samples", "geometry", "charfreq", "comparison", "dynamics", "length", "confusion", "robustness"]
    for w in which:
        globals()[f"fig_{w}"]()
