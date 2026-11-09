"""Split character-level errors into vowel-order (diacritic) and base-consonant confusion.

Ethiopic syllables occupy a regular grid in Unicode: for a codepoint ``cp`` in
``U+1200``-``U+1357``, ``(cp - 0x1200) // 8`` identifies the base consonant family
and ``(cp - 0x1200) % 8`` the vowel order.  A substitution that keeps the base and
changes the order is a *diacritic* error -- the model saw the right consonant and
the wrong vowel stroke.  A substitution that changes the base is a *base-character*
error.  Everything outside that range (Ethiopic wordspace and punctuation
``U+1360``-``U+1368``, numerals, the Ethiopic Supplement block, ASCII space) is not
on the grid and is counted separately.

Reuses ``align()`` from ``scripts/diacritic_confusion.py`` so the counts here are
produced by exactly the same alignment that generated ``tables/confusion_*.csv``.

Writes ``tables/error_modes.csv``.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from diacritic_confusion import align

ETH_LO, ETH_HI = 0x1200, 0x1357  # inclusive bounds of the 8-wide syllabic grid
ORDERS = [
    "1st (ge'ez)",
    "2nd (ka'eb)",
    "3rd (sales)",
    "4th (rabe')",
    "5th (hams)",
    "6th (sades)",
    "7th (sabe')",
    "8th (labialised)",
]

RUNS = [
    ("CRNN-VGG (best)", "abl_asp_48x512_s1"),
    ("CRNN-VGG seed 2", "abl_asp_48x512_s2"),
    ("CRNN-VGG seed 3", "abl_asp_48x512_s3"),
    ("CRNN-ConvNeXt", "crnn_convnext_s1"),
    ("SVTR-Small", "svtr_small_s1"),
    ("DiffOCR", "diff_ocr_s1"),
]


def grid(ch: str):
    """(base_index, vowel_order) for a syllabic character, else None."""
    cp = ord(ch)
    if ETH_LO <= cp <= ETH_HI:
        off = cp - ETH_LO
        return off // 8, off % 8
    return None


def analyse(run: str, split: str) -> dict:
    path = Path("results") / run / f"predictions_{split}.csv"
    counts = Counter()
    order_pairs = Counter()
    deleted = Counter()
    inserted = Counter()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for op, a, b in align(row["ground_truth"], row["prediction"]):
                if op == "=":
                    continue
                counts[op] += 1
                if op == "sub":
                    ga, gb = grid(a), grid(b)
                    if ga is None or gb is None:
                        counts["sub_nonsyll"] += 1
                    elif ga[0] == gb[0]:
                        counts["sub_diacritic"] += 1
                        order_pairs[(ga[1], gb[1])] += 1
                    else:
                        counts["sub_base"] += 1
                        if ga[1] == gb[1]:
                            counts["sub_base_same_order"] += 1
                elif op == "del":
                    deleted[a] += 1
                else:
                    inserted[b] += 1
    counts["edits"] = counts["sub"] + counts["ins"] + counts["del"]
    return {"counts": counts, "order_pairs": order_pairs,
            "deleted": deleted, "inserted": inserted}


def pct(x: int, total: int) -> float:
    return 100.0 * x / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("tables/error_modes.csv"), type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, run in RUNS:
        for split in ("test_I", "test_II"):
            r = analyse(run, split)
            c = r["counts"]
            e, s = c["edits"], c["sub"]
            rows.append({
                "model": label, "run": run, "split": split,
                "edits": e, "subs": s, "ins": c["ins"], "del": c["del"],
                "sub_diacritic": c["sub_diacritic"],
                "sub_base": c["sub_base"],
                "sub_nonsyll": c["sub_nonsyll"],
                "diacritic_pct_of_subs": round(pct(c["sub_diacritic"], s), 2),
                "base_pct_of_subs": round(pct(c["sub_base"], s), 2),
                "nonsyll_pct_of_subs": round(pct(c["sub_nonsyll"], s), 2),
                "diacritic_pct_of_edits": round(pct(c["sub_diacritic"], e), 2),
                "del_pct_of_edits": round(pct(c["del"], e), 2),
            })
            print(f"{label:18s} {split:8s} edits={e:7d} sub={s:6d} ins={c['ins']:6d} del={c['del']:6d}"
                  f"  diacritic={c['sub_diacritic']:6d} ({pct(c['sub_diacritic'], s):5.1f}% of subs)"
                  f"  base={c['sub_base']:6d} ({pct(c['sub_base'], s):5.1f}%)"
                  f"  non-syll={c['sub_nonsyll']:5d} ({pct(c['sub_nonsyll'], s):5.1f}%)")

            if run == "abl_asp_48x512_s1" and split == "test_II":
                d = c["sub_diacritic"]
                print("\n  top vowel-order confusions (best model, Test-II):")
                for (x, y), n in r["order_pairs"].most_common(6):
                    print(f"    {ORDERS[x]:18s} -> {ORDERS[y]:18s} {n:6d}  ({pct(n, d):5.1f}% of diacritic subs)")
                merged = Counter()
                for (x, y), n in r["order_pairs"].items():
                    merged[tuple(sorted((x, y)))] += n
                (x, y), n = merged.most_common(1)[0]
                print(f"    merged {ORDERS[x]} <-> {ORDERS[y]}: {n} ({pct(n, d):.1f}% of diacritic subs)")
                print(f"    base confusions keeping the same vowel order: "
                      f"{c['sub_base_same_order']} ({pct(c['sub_base_same_order'], c['sub_base']):.1f}% of base subs)")
                print("\n  top deletions:", ", ".join(
                    f"U+{ord(ch):04X}={n}" for ch, n in r["deleted"].most_common(4)))
                print("  top insertions:", ", ".join(
                    f"U+{ord(ch):04X}={n}" for ch, n in r["inserted"].most_common(4)), "\n")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
