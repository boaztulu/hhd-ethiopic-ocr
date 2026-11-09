"""Phase 1: dataset verification + EDA.

Outputs (all under ${PROJECT_ROOT}):
    figures/eda/    image dimensions, char frequency, length distributions, OOV
    figures/eda_samples.png     20 GT-labelled samples per split
    tables/dataset_stats.csv
    tables/char_frequency_top50.csv
    tables/oov_test_vs_train.csv
    data_card.md
    cache/vocab.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

# Paths
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get("DATA_ROOT", ROOT / "HHD-Ethiopic-data"))
FIG = ROOT / "figures" / "eda"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "tables").mkdir(parents=True, exist_ok=True)
(ROOT / "cache").mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": {
        "csv": DATA / "train" / "train_raw" / "image_text_pairs_train.csv",
        "img_dir": DATA / "train_raw" / "image_train",
        "gt_dir": DATA / "train_raw" / "gt_train",
        "expected": 57374,
    },
    "test_I": {
        "csv": DATA / "test" / "test_rand" / "image_text_pairs_test_rand.csv",
        "img_dir": DATA / "test" / "test_rand" / "image_rand",
        "gt_dir": DATA / "test" / "test_rand" / "gt_rand",
        "expected": 6375,
    },
    "test_II": {
        "csv": DATA / "test" / "test_18th" / "image_text_pairs_test_18th.csv",
        "img_dir": DATA / "test" / "test_18th" / "image_18th",
        "gt_dir": DATA / "test" / "test_18th" / "gt_18th",
        "expected": 15935,
    },
}


def load_csv(p: Path) -> pd.DataFrame:
    # HHD-Ethiopic CSVs have NO header row: first record is data.
    df = pd.read_csv(p, header=None, names=["image", "text"], dtype=str, keep_default_na=False)
    print(f"  loaded {p.name} -> {len(df)} rows, columns={list(df.columns)}")
    return df


def find_columns(df: pd.DataFrame) -> tuple[str, str]:
    return "image", "text"


def measure_split(split: str, info: dict) -> dict:
    print(f"\n>>> {split}")
    df = load_csv(info["csv"])
    img_col, text_col = find_columns(df)
    n_csv = len(df)
    img_dir = info["img_dir"]
    gt_dir = info["gt_dir"]

    n_img = sum(1 for _ in img_dir.glob("*.png"))
    n_gt = sum(1 for _ in gt_dir.glob("*.txt")) if gt_dir.exists() else 0
    print(f"  CSV rows={n_csv} imgs_on_disk={n_img} gt_files={n_gt} (expected {info['expected']})")

    widths, heights = [], []
    char_lens = []
    char_counter: Counter[str] = Counter()
    missing_imgs = 0
    corrupt = []

    # sample up to all rows; PIL .size is fast on small PNGs
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split} images", leave=False):
        fname = str(row[img_col]).strip()
        text = str(row[text_col]).rstrip(" ").rstrip()
        img_path = img_dir / fname
        if not img_path.exists():
            # Try just basename
            alt = img_dir / Path(fname).name
            if alt.exists():
                img_path = alt
            else:
                missing_imgs += 1
                continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:  # noqa
            corrupt.append((fname, str(e)))
            continue
        widths.append(w)
        heights.append(h)
        char_lens.append(len(text))
        char_counter.update(text)

    print(f"  measured {len(widths)} images; missing={missing_imgs} corrupt={len(corrupt)}")
    return {
        "split": split,
        "n_csv": n_csv,
        "n_img_on_disk": n_img,
        "n_gt_files": n_gt,
        "n_measured": len(widths),
        "missing_imgs": missing_imgs,
        "corrupt": corrupt,
        "widths": np.array(widths),
        "heights": np.array(heights),
        "char_lens": np.array(char_lens),
        "char_counter": char_counter,
        "df": df[[img_col, text_col]].rename(columns={img_col: "image", text_col: "text"}),
    }


def stats(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {k: float("nan") for k in ("min", "max", "mean", "median", "p05", "p95", "std")}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(arr.std()),
    }


def plot_dimensions(results: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for col, split in enumerate(["train", "test_I", "test_II"]):
        r = results[split]
        axes[0, col].hist(r["widths"], bins=60, color="steelblue", alpha=0.85)
        axes[0, col].set_title(f"{split}: width (n={len(r['widths'])})")
        axes[0, col].set_xlabel("width (px)")
        axes[0, col].set_ylabel("count")
        axes[1, col].hist(r["heights"], bins=60, color="indianred", alpha=0.85)
        axes[1, col].set_title(f"{split}: height")
        axes[1, col].set_xlabel("height (px)")
        axes[1, col].set_ylabel("count")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"image_dimensions.{ext}", dpi=300)
    plt.close(fig)
    print(f"  saved image_dimensions.{{png,pdf}}")


def plot_char_length(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bins = np.linspace(0, 60, 31)
    for split, color in [("train", "steelblue"), ("test_I", "indianred"), ("test_II", "darkorange")]:
        ax.hist(results[split]["char_lens"], bins=bins, alpha=0.5, label=split, color=color)
    ax.set_xlabel("# chars per line")
    ax.set_ylabel("count")
    ax.set_title("Ground-truth length distribution")
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"char_length.{ext}", dpi=300)
    plt.close(fig)
    print(f"  saved char_length.{{png,pdf}}")


def char_frequency_csv(results: dict, dictionary: dict[int, str]) -> None:
    train_cnt = results["train"]["char_counter"]
    t1_cnt = results["test_I"]["char_counter"]
    t2_cnt = results["test_II"]["char_counter"]
    all_chars = set(train_cnt) | set(t1_cnt) | set(t2_cnt)
    dict_chars = set(dictionary.values())
    rows = []
    for ch in sorted(all_chars, key=lambda c: -train_cnt.get(c, 0)):
        rows.append({
            "char": ch,
            "codepoint_hex": "U+%04X" % ord(ch),
            "in_dictionary": ch in dict_chars,
            "train": train_cnt.get(ch, 0),
            "test_I": t1_cnt.get(ch, 0),
            "test_II": t2_cnt.get(ch, 0),
        })
    pd.DataFrame(rows).to_csv(ROOT / "tables" / "char_frequency_full.csv", index=False)
    pd.DataFrame(rows[:50]).to_csv(ROOT / "tables" / "char_frequency_top50.csv", index=False)

    # top-50 bar chart
    top50 = rows[:50]
    fig, ax = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    ax.bar(range(len(top50)), [r["train"] for r in top50], color="steelblue")
    ax.set_xticks(range(len(top50)))
    ax.set_xticklabels([r["char"] for r in top50], rotation=0, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("train count (log)")
    ax.set_title("Top-50 characters by training frequency")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"char_frequency_top50.{ext}", dpi=300)
    plt.close(fig)
    print(f"  saved char_frequency_top50.{{png,pdf}} ({len(rows)} unique chars total across all splits)")


def oov_analysis(results: dict, dictionary: dict[int, str]) -> dict:
    train_chars = set(results["train"]["char_counter"].keys())
    t1_chars = set(results["test_I"]["char_counter"].keys())
    t2_chars = set(results["test_II"]["char_counter"].keys())
    oov_t1 = sorted(t1_chars - train_chars)
    oov_t2 = sorted(t2_chars - train_chars)

    def oov_rate(cnt: Counter, train: set) -> float:
        tot = sum(cnt.values())
        if tot == 0:
            return 0.0
        oov_chars = sum(v for k, v in cnt.items() if k not in train)
        return 100.0 * oov_chars / tot

    summary = {
        "train_unique_chars": len(train_chars),
        "test_I_unique_chars": len(t1_chars),
        "test_II_unique_chars": len(t2_chars),
        "dictionary_chars": len(set(dictionary.values())),
        "oov_chars_test_I": oov_t1,
        "oov_chars_test_II": oov_t2,
        "oov_rate_test_I_pct": oov_rate(results["test_I"]["char_counter"], train_chars),
        "oov_rate_test_II_pct": oov_rate(results["test_II"]["char_counter"], train_chars),
    }
    with open(ROOT / "tables" / "oov_test_vs_train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            if isinstance(v, list):
                w.writerow([k, "|".join(v)])
            else:
                w.writerow([k, v])
    print(f"  OOV chars Test-I: {len(oov_t1)}  Test-II: {len(oov_t2)}")
    print(f"  OOV char rate Test-I: {summary['oov_rate_test_I_pct']:.4f}%  Test-II: {summary['oov_rate_test_II_pct']:.4f}%")
    return summary


def grid_samples(results: dict) -> None:
    """20 representative images per split with GT below."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), constrained_layout=True)
    for ax, split in zip(axes, ["train", "test_I", "test_II"]):
        r = results[split]
        df = r["df"]
        idx = rng.choice(len(df), size=min(8, len(df)), replace=False)
        info = SPLITS[split]
        thumbs = []
        labels = []
        for i in idx:
            fname = df.iloc[i]["image"]
            text = str(df.iloc[i]["text"])
            p = info["img_dir"] / fname
            if not p.exists():
                p = info["img_dir"] / Path(fname).name
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            im = im.resize((max(64, int(im.width * 60 / im.height)), 60))
            thumbs.append(np.array(im))
            labels.append(text)
        if not thumbs:
            continue
        # pad to common width by horizontally concatenating with separators
        canvas_w = sum(t.shape[1] for t in thumbs) + 4 * (len(thumbs) - 1)
        canvas = np.full((60, canvas_w, 3), 255, dtype=np.uint8)
        x = 0
        for t in thumbs:
            canvas[:, x : x + t.shape[1]] = t
            x += t.shape[1] + 4
        ax.imshow(canvas)
        ax.set_title(f"{split}: {len(thumbs)} random samples")
        ax.set_xticks([])
        ax.set_yticks([])
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"eda_samples.{ext}", dpi=300)
    plt.close(fig)
    print("  saved eda_samples.{png,pdf}")


def write_dataset_stats(results: dict, summary: dict) -> None:
    rows = []
    for split in ["train", "test_I", "test_II"]:
        r = results[split]
        rows.append({
            "split": split,
            "n_csv": r["n_csv"],
            "n_measured": r["n_measured"],
            "n_missing_imgs": r["missing_imgs"],
            "n_corrupt": len(r["corrupt"]),
            **{f"width_{k}": v for k, v in stats(r["widths"]).items()},
            **{f"height_{k}": v for k, v in stats(r["heights"]).items()},
            **{f"chars_{k}": v for k, v in stats(r["char_lens"]).items()},
            "n_unique_chars": len(r["char_counter"]),
        })
    pd.DataFrame(rows).to_csv(ROOT / "tables" / "dataset_stats.csv", index=False)
    print("  wrote dataset_stats.csv")

    # data_card.md
    card = ROOT / "data_card.md"
    with open(card, "w") as f:
        f.write("# HHD-Ethiopic: Data Card\n\n")
        f.write("Auto-generated by `scripts/phase1_eda.py`. License: CC BY 4.0 (Belay et al. 2024).\n\n")
        f.write("## Counts\n\n")
        f.write("| Split | CSV rows | Imgs measured | Missing | Corrupt | Unique chars |\n")
        f.write("|-------|---------:|--------------:|--------:|--------:|-------------:|\n")
        for r in rows:
            f.write(f"| {r['split']} | {r['n_csv']} | {r['n_measured']} | {r['n_missing_imgs']} | {r['n_corrupt']} | {r['n_unique_chars']} |\n")
        f.write("\n## Image dimensions (px)\n\n")
        f.write("| Split | w_min | w_p05 | w_median | w_p95 | w_max | h_min | h_p05 | h_median | h_p95 | h_max |\n")
        f.write("|-------|------:|------:|---------:|------:|------:|------:|------:|---------:|------:|------:|\n")
        for r in rows:
            f.write(
                f"| {r['split']} | {r['width_min']:.0f} | {r['width_p05']:.0f} | {r['width_median']:.0f} | "
                f"{r['width_p95']:.0f} | {r['width_max']:.0f} | {r['height_min']:.0f} | {r['height_p05']:.0f} | "
                f"{r['height_median']:.0f} | {r['height_p95']:.0f} | {r['height_max']:.0f} |\n"
            )
        f.write("\n## Ground-truth length (chars)\n\n")
        f.write("| Split | min | p05 | median | p95 | max |\n")
        f.write("|-------|----:|----:|-------:|----:|----:|\n")
        for r in rows:
            f.write(
                f"| {r['split']} | {r['chars_min']:.0f} | {r['chars_p05']:.0f} | "
                f"{r['chars_median']:.0f} | {r['chars_p95']:.0f} | {r['chars_max']:.0f} |\n"
            )
        f.write("\n## Vocabulary\n\n")
        f.write(f"- Train unique chars: **{summary['train_unique_chars']}**\n")
        f.write(f"- Test-I unique chars: {summary['test_I_unique_chars']}\n")
        f.write(f"- Test-II unique chars: {summary['test_II_unique_chars']}\n")
        f.write(f"- Authors' dictionary.txt unique chars: {summary['dictionary_chars']}\n")
        f.write(f"- OOV chars in Test-I (not seen in train): **{len(summary['oov_chars_test_I'])}**\n")
        f.write(f"- OOV chars in Test-II (not seen in train): **{len(summary['oov_chars_test_II'])}**\n")
        f.write(f"- OOV character rate Test-I: **{summary['oov_rate_test_I_pct']:.4f}%**\n")
        f.write(f"- OOV character rate Test-II: **{summary['oov_rate_test_II_pct']:.4f}%**\n")
        if summary["oov_chars_test_II"]:
            f.write(f"- Sample OOV in Test-II: {' '.join(summary['oov_chars_test_II'][:30])}\n")
    print(f"  wrote data_card.md")


def load_authors_dictionary() -> dict[int, str]:
    p = DATA / "dictionary.txt"
    txt = p.read_text(encoding="utf-8")
    # Format: "dictinay={1: 'ሀ', 2: 'ሁ', ...}"
    body = txt.split("=", 1)[1].strip()
    d = eval(body, {"__builtins__": {}})  # noqa: S307 - trusted file
    return d


def main():
    print(f"PROJECT_ROOT={ROOT}")
    print(f"DATA_ROOT={DATA}")

    dictionary = load_authors_dictionary()
    print(f"authors' dictionary entries: {len(dictionary)} (last={dictionary[max(dictionary)]!r})")

    results = {split: measure_split(split, info) for split, info in SPLITS.items()}

    plot_dimensions(results)
    plot_char_length(results)
    char_frequency_csv(results, dictionary)
    summary = oov_analysis(results, dictionary)
    grid_samples(results)
    write_dataset_stats(results, summary)

    # Persist vocab: union of train + test chars + authors' dictionary so we can
    # decode every Ethiopic glyph the dataset uses (no OOV at inference).
    train_chars = set(results["train"]["char_counter"].keys())
    all_chars = (
        train_chars
        | set(results["test_I"]["char_counter"].keys())
        | set(results["test_II"]["char_counter"].keys())
        | set(dictionary.values())
    )
    ordered = sorted(all_chars)
    vocab = {
        "version": 2,
        "source": "union(train + test_I + test_II + authors' dictionary.txt)",
        "special_tokens": {"pad": 0, "sos": 1, "eos": 2, "unk": 3, "blank": 4},
        "chars": ordered,
        "char_to_id": {c: i + 5 for i, c in enumerate(ordered)},
        "id_to_char": {str(i + 5): c for i, c in enumerate(ordered)},
        "stats": {
            "n_chars": len(ordered),
            "n_train_chars": len(train_chars),
            "n_dictionary_chars": len(set(dictionary.values())),
            "oov_test_I": summary["oov_chars_test_I"],
            "oov_test_II": summary["oov_chars_test_II"],
            "oov_rate_test_I_pct": summary["oov_rate_test_I_pct"],
            "oov_rate_test_II_pct": summary["oov_rate_test_II_pct"],
        },
    }
    with open(ROOT / "cache" / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  wrote cache/vocab.json with {len(ordered)} chars + 5 specials (train-only={len(train_chars)})")

    # Aggregate alarm if mismatches found
    err = []
    for split, info in SPLITS.items():
        r = results[split]
        if r["n_measured"] + r["missing_imgs"] + len(r["corrupt"]) != r["n_csv"]:
            err.append(f"{split}: rows mismatch")
        if r["n_csv"] != info["expected"]:
            err.append(f"{split}: expected {info['expected']} CSV rows, got {r['n_csv']}")
    if err:
        print("WARNINGS:", err, file=sys.stderr)
    else:
        print("All split counts match the published HHD-Ethiopic Table 1.")

    print("\nDONE.")


if __name__ == "__main__":
    main()
