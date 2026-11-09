"""Where do the geometry and augmentation gains occur, by line aspect ratio?

The 48x368 stretch distorts lines in proportion to how far their aspect ratio
is from 368:48 (narrow lines stretched, wide lines compressed); the 48x512
aspect-preserving canvas leaves lines up to 512:48 undistorted. A gain that
comes from geometry should therefore concentrate in the distorted bins, while
an augmentation gain has no reason to depend on aspect ratio.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # project root
OUT = ROOT / "results" / "analysis"
G = {
    "asp_medium": ["abl_asp_48x512_s1", "abl_asp_48x512_s2", "abl_asp_48x512_s3"],
    "368_none": ["abl_aug_none_s1", "abl_pub_48x368_s1"],
    "368_light": ["crnn_vgg_s1", "crnn_vgg_s2_reeval", "crnn_vgg_s3"],
    "368_heavy": ["abl_aug_heavy_s1"],
}
EDGES = [0, 5, 368 / 48, 512 / 48, 14, 1e9]
NAMES = ["<5", "5-7.7", "7.7-10.7", "10.7-14", ">14"]
rng = np.random.default_rng(7)
res = {}
for split in ("test_I", "test_II"):
    g = np.load(OUT / f"geom_{split}.npz"); A = g["W"] / g["H"]
    ed = {r: np.load(OUT / f"ed_{r}_{split}.npz")["ed"].astype(float) for rs in G.values() for r in rs}
    L = np.load(OUT / f"ed_{G['asp_medium'][0]}_{split}.npz")["L"].astype(float)
    b = np.digitize(A, EDGES[1:-1])
    rows = []
    for k, name in enumerate(NAMES):
        m = b == k
        cer = {gn: float(np.mean([100 * ed[r][m].sum() / L[m].sum() for r in rs])) for gn, rs in G.items()}

        def ci(a, c):
            idx = np.flatnonzero(m); d = []
            for _ in range(1000):
                s = rng.choice(idx, size=idx.size)
                f = lambda gn: np.mean([100 * ed[r][s].sum() / L[s].sum() for r in G[gn]])
                d.append(f(c) - f(a))
            return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]
        row = {"bin": name, "n": int(m.sum()), "frac": round(float(m.mean() * 100), 1),
               **{f"cer_{gn}": round(v, 2) for gn, v in cer.items()},
               "d_geom_plus_med": round(cer["asp_medium"] - cer["368_light"], 2),
               "d_geom_plus_med_ci": ci("368_light", "asp_medium"),
               "d_aug_none_light": round(cer["368_light"] - cer["368_none"], 2),
               "d_aug_light_heavy": round(cer["368_heavy"] - cer["368_light"], 2),
               "d_aug_light_heavy_ci": ci("368_light", "368_heavy")}
        rows.append(row)
        print(split, json.dumps(row), flush=True)
    res[split] = rows
json.dump(res, open(OUT / "aspect_strat.json", "w"), indent=2)
