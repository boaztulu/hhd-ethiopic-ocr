"""Re-score every run under the benchmark's own CER procedure and under a per-line mean.

1. Decode the benchmark's label arrays (y_test_*.npy, 0-padded ids into dictionary.txt)
   and check they equal our CSV ground truth line by line.
2. Recompute CER exactly as Test_HPopt-Attn-CTC.ipynb does: edit distance on id
   sequences, summed, over the total number of ground-truth ids (corpus-level, Eq. 3).
3. Also report the mean of per-line CERs (a common alternative definition) and NED.
Per-line edit distances are saved for the bootstrap analyses.
"""
import ast, csv, io, json, zipfile
from pathlib import Path
import numpy as np
import editdistance

ROOT = Path(__file__).resolve().parents[2]  # project root
OUT = ROOT / "results" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)

txt = (ROOT / "HHD-Ethiopic-data/dictionary.txt").read_text(encoding="utf-8")
id2ch = ast.literal_eval(txt[txt.index("{"):])
ch2id = {c: i for i, c in id2ch.items()}

NPY = {"test_I": (ROOT / "HHD-Ethiopic-data/test/test_rand/test_rand_numpy.zip", "test_rand_numpy/y_test_rand.npy"),
       "test_II": (ROOT / "HHD-Ethiopic-data/test/test_18th/test_18th_numpy.zip", "test_18th_numpy/y_test_18th.npy")}
RUNS = ["abl_asp_48x512_s1", "abl_asp_48x512_s2", "abl_asp_48x512_s3",
        "crnn_vgg_s1", "crnn_vgg_s2_reeval", "crnn_vgg_s3",
        "abl_aug_none_s1", "abl_pub_48x368_s1", "abl_aug_heavy_s1",
        "svtr_small_s1", "diff_ocr_s1", "crnn_convnext_s1"]


def load_preds(run, split):
    with open(ROOT / f"results/{run}/predictions_{split}.csv", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        return [(a, b, c) for a, b, c in r]


report = {}
for split in ("test_I", "test_II"):
    y = np.load(io.BytesIO(zipfile.ZipFile(NPY[split][0]).read(NPY[split][1])))
    gt_npy = ["".join(id2ch[int(v)] for v in row if v != 0) for row in y]
    ref = load_preds(RUNS[0], split)
    gt_csv = [g for _, g, _ in ref]
    mism = sum(a != b for a, b in zip(gt_npy, gt_csv))
    total_ids = int((y != 0).sum())
    report[f"{split}_gt_mismatches"] = mism
    report[f"{split}_total_gt_ids"] = total_ids
    report[f"{split}_total_gt_chars_csv"] = sum(len(g) for g in gt_csv)
    print(split, "npy-vs-csv GT mismatches:", mism, " total gt ids:", total_ids, flush=True)
    true_ids = [[int(v) for v in row if v != 0] for row in y]
    for run in RUNS:
        rows = load_preds(run, split)
        assert [g for _, g, _ in rows] == gt_csv, f"order/GT mismatch in {run}"
        preds = [p for _, _, p in rows]
        # benchmark notebook procedure on id sequences; unknown chars get a unique id
        pred_ids = [[ch2id.get(c, -2) for c in p] for p in preds]
        ed_ids = np.array([editdistance.eval(t, p) for t, p in zip(true_ids, pred_ids)])
        ed = np.array([editdistance.eval(g, p) for g, p in zip(gt_csv, preds)])
        L = np.array([len(g) for g in gt_csv]); P = np.array([len(p) for p in preds])
        np.savez(OUT / f"ed_{run}_{split}.npz", ed=ed, L=L, P=P)
        r = {
            "cer_benchmark_procedure": 100 * ed_ids.sum() / total_ids,
            "cer_ours": 100 * ed.sum() / L.sum(),
            "cer_per_line_mean": 100 * float(np.mean(ed / np.maximum(L, 1))),
            "ned": 100 * float(np.mean(ed / np.maximum(np.maximum(L, P), 1))),
        }
        report[f"{run}|{split}"] = {k: round(v, 3) for k, v in r.items()}
        print(f"  {run:22s} " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)

json.dump(report, open(OUT / "rescore.json", "w"), indent=2)
