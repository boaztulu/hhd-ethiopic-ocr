"""One-shot summary: text snapshot of all runs + best-by-T2 winner highlighted.

Run on the login node any time (no GPU required) to see where we stand.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PUBLISHED = {
    "Plain-CTC":          (20.88, 33.56),
    "Attn-CTC":           (19.42, 33.07),
    "HPopt-Plain-CTC":    (19.42, 32.01),
    "HPopt-Attn-CTC":     (16.41, 28.65),
    "CRNN (Shi 2016)":    (21.04, 29.86),
    "ASTER":              (24.43, 35.13),
    "SVTR (Du 2022)":     (19.78, 30.82),
    "ABINet":             (21.49, 32.76),
    "best human":         (25.39, 33.20),
    "avg human":          (30.46, 35.63),
}

BAR = "HPopt-Attn-CTC"


def best_row(rows: list[dict]) -> dict | None:
    rows = [r for r in rows if r.get("val_cer") is not None]
    if not rows:
        return None
    return min(rows, key=lambda r: r["val_cer"])


def main():
    print("=" * 76)
    print(f"HHD-Ethiopic: live results vs published {BAR} (T1={PUBLISHED[BAR][0]} / T2={PUBLISHED[BAR][1]})")
    print("=" * 76)
    print()
    print(f"{'run':35s} {'epoch':>6s} {'val':>7s} {'T1 CER':>9s} {'T2 CER':>9s} {'ΔT1':>7s} {'ΔT2':>7s}")
    print("-" * 76)

    runs = []
    for p in sorted((ROOT / "checkpoints").glob("*/metrics.jsonl")):
        rows = []
        try:
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        except Exception:
            continue
        b = best_row(rows)
        if b is None:
            continue
        runs.append((p.parent.name, b))

    runs.sort(key=lambda r: (r[1].get("test_II_cer") or 1e9))

    pub_t1, pub_t2 = PUBLISHED[BAR]
    for name, b in runs:
        t1 = b.get("test_I_cer", float("nan"))
        t2 = b.get("test_II_cer", float("nan"))
        dt1 = t1 - pub_t1 if t1 is not None else float("nan")
        dt2 = t2 - pub_t2 if t2 is not None else float("nan")
        win = " ←" if (t1 is not None and t2 is not None and t1 < pub_t1 and t2 < pub_t2) else ""
        print(f"{name:35s} {b['epoch']:>6d} {b['val_cer']:>7.2f} {t1:>9.2f} {t2:>9.2f} {dt1:>+7.2f} {dt2:>+7.2f}{win}")

    print("-" * 76)
    print("Published baselines (Belay 2024, Table 3):")
    print()
    print(f"{'method':35s} {'T1 CER':>9s} {'T2 CER':>9s}")
    for name, (c1, c2) in PUBLISHED.items():
        marker = " ←bar" if name == BAR else ""
        print(f"{name:35s} {c1:>9.2f} {c2:>9.2f}{marker}")


if __name__ == "__main__":
    main()
