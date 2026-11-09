"""Best-validation checkpoint vs last epoch, and late-schedule slopes, from metrics.jsonl.

Shows which epoch each reported number comes from (Table 4 = offline eval of
best.pt; Table 6 = per-epoch log at epochs 25 and 80) and how much each family
was still improving near the end of the 80-epoch cosine schedule.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root
OUT = ROOT / "results" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
RUNS = {"abl_asp_48x512_s1": "abl_asp_48x512_s1", "abl_asp_48x512_s2": "abl_asp_48x512_s2",
        "abl_asp_48x512_s3": "abl_asp_48x512_s3", "crnn_vgg_s1": "crnn_vgg_s1",
        "crnn_vgg_s2": "crnn_vgg_s2_reeval", "crnn_vgg_s3": "crnn_vgg_s3",
        "abl_aug_none_s1": "abl_aug_none_s1", "abl_pub_48x368_s1": "abl_pub_48x368_s1",
        "abl_aug_heavy_s1": "abl_aug_heavy_s1", "svtr_small_s1": "svtr_small_s1",
        "diff_ocr_s1": "diff_ocr_s1", "crnn_convnext_s1": "crnn_convnext_s1"}

res = {}
for run, resdir in RUNS.items():
    rows = [json.loads(l) for l in open(ROOT / f"checkpoints/{run}/metrics.jsonl") if l.strip()]
    by_ep = {r["epoch"]: r for r in rows}
    eps = sorted(by_ep)
    best = min(rows, key=lambda r: r["val_cer"])
    off = json.load(open(ROOT / f"results/{resdir}/metrics.json"))
    g = lambda e, k: by_ep[e].get(k) if e in by_ep else None
    r = {
        "n_epochs_logged": len(eps), "last_epoch": eps[-1],
        "best_epoch": best["epoch"], "best_val_cer": round(best["val_cer"], 3),
        "best_T1_logged": round(best.get("test_I_cer", float("nan")), 3),
        "best_T2_logged": round(best.get("test_II_cer", float("nan")), 3),
        "offline_T1": round(off["test_I"]["cer"], 3), "offline_T2": round(off["test_II"]["cer"], 3),
        "ep24_T1": g(24, "test_I_cer"), "ep24_T2": g(24, "test_II_cer"), "ep25_T2": g(25, "test_II_cer"),
        "last_T1": round(by_ep[eps[-1]]["test_I_cer"], 3), "last_T2": round(by_ep[eps[-1]]["test_II_cer"], 3),
        "last_val": round(by_ep[eps[-1]]["val_cer"], 3),
        "train_hours": round(sum(r_["epoch_time_s"] for r_ in rows) / 3600, 2),
    }
    for a, b in ((59, 79), (69, 79), (49, 79)):
        # DiffOCR only scored the test sets every 5 epochs; fall back to the nearest logged one
        ta = next((e for e in (a, a + 1) if "test_II_cer" in by_ep.get(e, {})), None)
        if ta is not None and "test_II_cer" in by_ep[b]:
            r[f"dT2_{ta+1}to{b+1}"] = round(by_ep[b]["test_II_cer"] - by_ep[ta]["test_II_cer"], 3)
        r[f"dval_{a+1}to{b+1}"] = round(by_ep[b]["val_cer"] - by_ep[a]["val_cer"], 3)
    # first epoch below the published Test-II bound
    cross = [e for e in eps if by_ep[e].get("test_II_cer", 99) < 28.65]
    r["first_epoch_below_28.65"] = (cross[0] + 1) if cross else None
    res[run] = r
    print(run, json.dumps(r), flush=True)

json.dump(res, open(OUT / "logs.json", "w"), indent=2)
