"""Per-seed CER of the best configuration with 95% bootstrap intervals, and the seed spread.

Same procedure as bootstrap.py: 2,000 resamples of the per-line edit distances, single lines
on Test-I (stored in random order) and, on Test-II (stored in source order), single lines and
circular moving blocks of 50 consecutive lines as a proxy for pages. Also reports the sample
standard deviation of the unrounded per-seed CERs and the pooled seed standard deviation of the
configurations that have repeated runs, which the ablation intervals do not include.
"""
import json
import statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # project root
OUT = ROOT / "results" / "analysis"
B = 2000
rng = np.random.default_rng(20261118)

SEEDS = ["abl_asp_48x512_s1", "abl_asp_48x512_s2", "abl_asp_48x512_s3"]
REPLICATED = {  # configurations with more than one run (Table 8)
    "asp_medium": SEEDS,
    "368_light": ["crnn_vgg_s1", "crnn_vgg_s2_reeval", "crnn_vgg_s3"],
    "368_none": ["abl_aug_none_s1", "abl_pub_48x368_s1"],
}


def load(run, split):
    z = np.load(OUT / f"ed_{run}_{split}.npz")
    return z["ed"].astype(float), z["L"].astype(float)


def block_indices(n, b, rng):
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=k)
    idx = (starts[:, None] + np.arange(b)[None, :]) % n
    return idx.ravel()[:n]


def cer(ed, L):
    return 100 * ed.sum() / L.sum()


res = {"per_seed": {}, "spread": {}}
for split, blocks in (("test_I", [1]), ("test_II", [1, 50])):
    for run in SEEDS:
        ed, L = load(run, split)
        row = {"cer": round(cer(ed, L), 6)}
        for b in blocks:
            vals = []
            for _ in range(B):
                idx = block_indices(len(L), b, rng)
                vals.append(cer(ed[idx], L[idx]))
            row[f"ci_block{b}"] = [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
        res["per_seed"].setdefault(split, {})[run] = row
        print(split, run, json.dumps(row), flush=True)
    sds, dfs = {}, {}
    for name, runs in REPLICATED.items():
        v = [cer(*load(r, split)) for r in runs]
        sds[name] = {"values": [round(x, 6) for x in v], "mean": round(st.mean(v), 6),
                     "sample_sd": round(st.stdev(v), 4), "sd_of_2dp_values": round(st.stdev([round(x, 2) for x in v]), 4),
                     "population_sd": round(st.pstdev(v), 4)}
        dfs[name] = (len(v) - 1, st.variance(v))
    pooled = (sum(d * var for d, var in dfs.values()) / sum(d for d, _ in dfs.values())) ** 0.5
    res["spread"][split] = {"by_config": sds, "pooled_seed_sd": round(pooled, 4)}
    print(split, "spread:", json.dumps(res["spread"][split]), flush=True)

json.dump(res, open(OUT / "per_seed_ci.json", "w"), indent=2)
