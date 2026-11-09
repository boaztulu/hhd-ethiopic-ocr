"""Collect *current best* metrics from every `checkpoints/*/metrics.jsonl`
and write them as ``results/<run>/metrics_running.json``.

Used to populate `tables/main_results.csv` mid-training so we can preview the
comparison figure without waiting for a full eval pass.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def best_row(rows: list[dict]) -> dict | None:
    rows = [r for r in rows if r.get("val_cer") is not None]
    if not rows:
        return None
    return min(rows, key=lambda r: r["val_cer"])


def main() -> None:
    n_written = 0
    for p in sorted((ROOT / "checkpoints").glob("*/metrics.jsonl")):
        run = p.parent.name
        try:
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        except Exception as e:
            print(f"skip {run}: {e}")
            continue
        best = best_row(rows)
        if best is None:
            continue
        cfg = {}
        cfg_path = p.parent / "config_used.yaml"
        if cfg_path.exists():
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text())
        params_M = None
        # rough estimate via state dict size if present
        out_dir = ROOT / "results" / run
        out_dir.mkdir(parents=True, exist_ok=True)
        # Don't overwrite a real eval (one with bootstrap CIs). Write a
        # separate snapshot file instead.
        target_metrics = out_dir / "metrics.json"
        is_real_eval = False
        if target_metrics.exists():
            try:
                existing = json.loads(target_metrics.read_text())
                if existing.get("test_I", {}).get("cer_ci95") is not None:
                    is_real_eval = True
            except Exception:
                pass
        write_path = out_dir / ("metrics_snapshot.json" if is_real_eval else "metrics.json")
        snapshot = {
            "test_I": {
                "cer": best.get("test_I_cer"),
                "ned": best.get("test_I_ned"),
                "wer": best.get("test_I_wer"),
                "seq_acc": best.get("test_I_seq_acc"),
            },
            "test_II": {
                "cer": best.get("test_II_cer"),
                "ned": best.get("test_II_ned"),
                "wer": best.get("test_II_wer"),
                "seq_acc": best.get("test_II_seq_acc"),
            },
            "params_M": params_M,
            "best_val_cer": best["val_cer"],
            "best_epoch": best["epoch"],
            "ckpt": str(p.parent / "last.pt"),
            "config": cfg,
        }
        write_path.write_text(json.dumps(snapshot, indent=2, default=float))
        n_written += 1
        t1 = best.get("test_I_cer")
        t2 = best.get("test_II_cer")
        t1s = f"{t1:.2f}" if isinstance(t1, (int, float)) else "n/a"
        t2s = f"{t2:.2f}" if isinstance(t2, (int, float)) else "n/a"
        print(f"  {run:30s}  val={best['val_cer']:.2f}  T1={t1s}  T2={t2s}")
    print(f"wrote {n_written} snapshots → results/*/metrics.json")


if __name__ == "__main__":
    main()
