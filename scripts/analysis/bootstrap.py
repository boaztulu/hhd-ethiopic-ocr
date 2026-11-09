"""Dependence-robust bootstrap intervals for corpus CER.

Page identifiers are not distributed with HHD-Ethiopic. Test-II is stored in
source order (lag-1 autocorrelation of line height 0.85), so a circular
moving-block bootstrap over consecutive lines approximates resampling pages;
growing the block from one line to many pages shows how much line-level
resampling understates the interval. Test-I is stored in random order
(autocorrelation ~0), so only line-level resampling is available there.
Paired contrasts reuse the same resampled indices for every run.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # project root
OUT = ROOT / "results" / "analysis"
B = 2000
rng = np.random.default_rng(20261102)

GROUPS = {
    "asp_medium": ["abl_asp_48x512_s1", "abl_asp_48x512_s2", "abl_asp_48x512_s3"],
    "368_none": ["abl_aug_none_s1", "abl_pub_48x368_s1"],
    "368_light": ["crnn_vgg_s1", "crnn_vgg_s2_reeval", "crnn_vgg_s3"],
    "368_heavy": ["abl_aug_heavy_s1"],
    "svtr": ["svtr_small_s1"],
    "diffocr": ["diff_ocr_s1"],
}


def load(run, split):
    z = np.load(OUT / f"ed_{run}_{split}.npz")
    return z["ed"].astype(float), z["L"].astype(float)


def block_indices(n, b, rng):
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=k)
    idx = (starts[:, None] + np.arange(b)[None, :]) % n
    return idx.ravel()[:n]


res = {}
for split, blocks in (("test_II", [1, 10, 25, 50, 100, 200, 400, 800]), ("test_I", [1])):
    ed = {r: load(r, split)[0] for g in GROUPS.values() for r in g}
    L = load(GROUPS["asp_medium"][0], split)[1]
    n = len(L)
    point = {g: float(np.mean([100 * ed[r].sum() / L.sum() for r in rs])) for g, rs in GROUPS.items()}
    res[split] = {"point": point, "by_block": {}}
    for b in blocks:
        samples = {g: [] for g in GROUPS}
        best_seed = []
        for _ in range(B):
            idx = block_indices(n, b, rng)
            Ls = L[idx].sum()
            for g, rs in GROUPS.items():
                samples[g].append(np.mean([100 * ed[r][idx].sum() / Ls for r in rs]))
            best_seed.append(100 * ed["abl_asp_48x512_s2"][idx].sum() / Ls)
        S = {g: np.array(v) for g, v in samples.items()}
        ci = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
        row = {
            "best_seed_ci": ci(np.array(best_seed)),
            "asp_medium_mean_ci": ci(S["asp_medium"]),
            "svtr_ci": ci(S["svtr"]), "diffocr_ci": ci(S["diffocr"]),
            "d_none_to_light": ci(S["368_light"] - S["368_none"]),
            "d_light_to_heavy": ci(S["368_heavy"] - S["368_light"]),
            "d_light368_to_aspmedium": ci(S["asp_medium"] - S["368_light"]),
            "d_aspmedium_vs_diffocr": ci(S["diffocr"] - S["asp_medium"]),
            "d_aspmedium_vs_svtr": ci(S["svtr"] - S["asp_medium"]),
            "d_svtr_vs_diffocr": ci(S["diffocr"] - S["svtr"]),
        }
        row["best_seed_halfwidth"] = round((row["best_seed_ci"][1] - row["best_seed_ci"][0]) / 2, 3)
        res[split]["by_block"][str(b)] = row
        print(split, f"b={b:4d}", json.dumps(row), flush=True)
    print(split, "points:", {k: round(v, 3) for k, v in point.items()}, flush=True)

json.dump(res, open(OUT / "bootstrap.json", "w"), indent=2)
