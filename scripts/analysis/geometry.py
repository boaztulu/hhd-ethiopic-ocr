"""Line-image geometry, canvas overflow, CTC length budgets and storage order.

Answers: how many lines overflow the 368:48 and 512:48 canvases, what happens
to them under each resizing rule, how many training lines the published mmocr
CRNN (T=26 at 32x100) and our ConvNeXt (T=16) cannot align, and whether the
distributed file order groups lines by source page.
"""
import csv, io, json, sys, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # project root
D = ROOT / "HHD-Ethiopic-data"
# Images are read from local copies of the distributed zips: per-file reads from
# a cloud-synced folder take ~0.25 s per PNG.
ZIPS = ROOT / "HHD-Ethiopic-data"
SPLITS = {
    "train": (D / "train/train_raw/image_text_pairs_train.csv", ZIPS / "train_raw.zip", "train_raw/image_train/"),
    "test_I": (D / "test/test_rand/image_text_pairs_test_rand.csv", ZIPS / "test.zip", "test/test_rand/image_rand/"),
    "test_II": (D / "test/test_18th/image_text_pairs_test_18th.csv", ZIPS / "test.zip", "test/test_18th/image_18th/"),
}
OUT = ROOT / "results" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)


def read_csv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 2:
                rows.append((r[0].strip(), r[1].rstrip(" ").rstrip()))
    return rows


def ctc_min_frames(s):
    """Minimum CTC input length for target s: len + number of adjacent repeats."""
    return len(s) + sum(1 for a, b in zip(s, s[1:]) if a == b)


def lag1(x):
    x = np.asarray(x, float); x = x - x.mean()
    return float((x[:-1] * x[1:]).sum() / (x * x).sum())


res = {}
for split, (csvp, zpath, prefix) in SPLITS.items():
    rows = read_csv(csvp)
    W, H = [], []
    with zipfile.ZipFile(zpath) as zf:
        for fn, _ in rows:
            with Image.open(io.BytesIO(zf.read(prefix + fn))) as im:
                w, h = im.size
            W.append(w); H.append(h)
    W = np.array(W); H = np.array(H); A = W / H
    L = np.array([len(t) for _, t in rows])
    F = np.array([ctc_min_frames(t) for _, t in rows])
    np.savez(OUT / f"geom_{split}.npz", W=W, H=H, L=L, F=F)
    r = {"n": len(rows)}
    r["aspect_pct"] = {p: round(float(np.percentile(A, p)), 2) for p in (5, 25, 50, 75, 90, 95, 99)}
    r["width_pct"] = {p: int(np.percentile(W, p)) for p in (50, 95)}
    r["height_pct"] = {p: int(np.percentile(H, p)) for p in (5, 50, 95)}
    for name, ratio in (("368", 368 / 48), ("512", 512 / 48)):
        over = A > ratio
        r[f"frac_over_{name}"] = round(float(over.mean() * 100), 2)
        if over.any():
            # horizontal compression relative to an aspect-preserving 48-px render
            r[f"squeeze_{name}_median"] = round(float(np.median(A[over] / ratio)), 2)
            r[f"squeeze_{name}_p95"] = round(float(np.percentile(A[over] / ratio, 95)), 2)
    # Belay letterbox: lines wider than 368:48 are scaled to fit the width, so the
    # ink band keeps only 48 * (368/48) / aspect = 368 / aspect pixels of height.
    over = A > 368 / 48
    r["letterbox_text_height_px_median_over"] = round(float(np.median(368 / A[over])), 1)
    r["letterbox_text_height_px_median_all"] = round(float(np.median(np.minimum(48, 368 / A))), 1)
    # stretch resize to 48x368: horizontal scale vs aspect-preserving 48-px render
    hs = 368 / (A * 48)
    r["stretch368_frac_compressed"] = round(float((hs < 1).mean() * 100), 2)
    r["stretch368_frac_stretched_gt1.5"] = round(float((hs > 1.5).mean() * 100), 2)
    for T in (16, 26, 45, 91, 127):
        r[f"frac_unalignable_T{T}"] = round(float((F > T).mean() * 100), 2)
    r["len_pct"] = {p: int(np.percentile(L, p)) for p in (50, 95)}
    r["len_max"] = int(L.max())
    r["lag1_height"] = round(lag1(H), 3)
    r["lag1_width"] = round(lag1(W), 3)
    r["lag1_aspect"] = round(lag1(A), 3)
    res[split] = r
    print(split, json.dumps(r, ensure_ascii=False), flush=True)

json.dump(res, open(OUT / "geometry.json", "w"), indent=2)
