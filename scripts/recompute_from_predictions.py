"""Recompute metrics.json from existing predictions_test_*.csv files.

Use when the metrics.json got clobbered or when you change the metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import bootstrap_ci, compute_metrics


def recompute(run_dir: Path, n_boot: int = 1000) -> dict:
    out = {}
    for split in ("test_I", "test_II"):
        p = run_dir / f"predictions_{split}.csv"
        if not p.exists():
            continue
        preds, gts = [], []
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                preds.append(r["prediction"])
                gts.append(r["ground_truth"])
        m = compute_metrics(preds, gts)
        cer_lo, cer_hi = bootstrap_ci(preds, gts, "cer", n_boot=n_boot)
        ned_lo, ned_hi = bootstrap_ci(preds, gts, "ned", n_boot=n_boot)
        out[split] = {
            "cer": m.cer, "ned": m.ned, "wer": m.wer, "seq_acc": m.seq_acc, "n": m.n,
            "cer_ci95": [cer_lo, cer_hi], "ned_ci95": [ned_lo, ned_hi],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    for d in args.run_dirs:
        if not d.is_dir():
            continue
        existing = {}
        meta_p = d / "metrics.json"
        if meta_p.exists():
            try:
                existing = json.loads(meta_p.read_text())
            except Exception:
                pass
        # preserve any non-test fields
        out = dict(existing)
        new = recompute(d, n_boot=args.bootstrap)
        out.update(new)
        if new:
            meta_p.write_text(json.dumps(out, indent=2, default=float))
            for split, v in new.items():
                print(f"{d.name}  {split}: CER={v['cer']:.2f}  CI={v['cer_ci95']}  WER={v['wer']:.2f}")
        else:
            print(f"{d.name}: no predictions found")


if __name__ == "__main__":
    main()
