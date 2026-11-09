"""Aggregate every model's eval JSON into the main results CSV + LaTeX table.

Scans ``results/*/metrics.json`` and produces:
  * tables/main_results.csv   (raw, machine-readable; keeps internal run names)
  * tables/main_results.tex   (publication table; human-readable display names)
  * tables/main_results.md     (raw, internal)

The published HHD-Ethiopic numbers (Belay et al. 2024) are always included as
the top rows for direct comparison. The LaTeX table uses human-readable display
names (no raw run identifiers / underscores) and adds a mean+/-std headline row
for the aspect-preserved CRNN-VGG (our best model, three seeds).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


PUBLISHED = [
    # (method, params_M, T1 CER, T1 NED, T2 CER, T2 NED)
    ("Plain-CTC (Belay 2020a)",    2.5, 20.88, 19.09, 33.56, 31.90),
    ("Attn-CTC (Belay 2021a)",     1.9, 19.42, 21.01, 33.07, 32.92),
    ("HPopt-Plain-CTC",            4.5, 19.42, 17.77, 32.01, 29.02),
    ("HPopt-Attn-CTC (prior best)", 2.2, 16.41, 16.06, 28.65, 27.37),
    ("CRNN (Shi 2016)",            8.3, 21.04, 21.01, 29.86, 29.29),
    ("ASTER (Shi 2018)",          27.0, 24.43, 20.88, 35.13, 30.75),
    ("SVTR (Du 2022)",             6.0, 19.78, 17.98, 30.82, 28.00),
    ("ABINet (Fang 2021)",        23.0, 21.49, 18.11, 32.76, 28.84),
    ("Best human",                  None, 25.39, 23.78, 33.20, 30.77),
    ("Avg human",                   None, 30.46, 26.32, 35.63, 38.59),
]

# Human-readable display names for the LaTeX table, keyed by results/<run> dir.
# "asp." = aspect-preserved 48x512 input; bare geometry = 48x368 stretch (the published input size).
DISPLAY = {
    "abl_asp_48x512_s1": r"CRNN-VGG, $48\times512$ asp.\ (seed 1)",
    "abl_asp_48x512_s2": r"CRNN-VGG, $48\times512$ asp.\ (seed 2)",
    "abl_asp_48x512_s3": r"CRNN-VGG, $48\times512$ asp.\ (seed 3)",
    "crnn_vgg_s1":       r"CRNN-VGG, $48\times368$ (seed 1)",
    "crnn_vgg_s2":       r"CRNN-VGG, $48\times368$ (seed 2)",
    "crnn_vgg_s3":       r"CRNN-VGG, $48\times368$ (seed 3)",
    "abl_aug_none_s1":   r"CRNN-VGG, $48\times368$, no aug.",
    "abl_pub_48x368_s1": r"CRNN-VGG, $48\times368$, no aug.\ (run B)",
    "abl_aug_heavy_s1":  r"CRNN-VGG, $48\times368$, heavy aug.",
    "crnn_convnext_s1":  r"CRNN-ConvNeXt, $48\times512$ asp.",
    "svtr_small_s1":     r"SVTR-Small, $48\times512$ asp.",
    "diff_ocr_s1":       r"DiffOCR, $48\times512$ asp.",
    "parseq_small_s1":   r"PARSeq-Small (did not converge)",
}

# Order our rows appear in the LaTeX table.
ORDER = [
    "abl_asp_48x512_s1", "abl_asp_48x512_s2", "abl_asp_48x512_s3",
    "crnn_vgg_s1", "crnn_vgg_s2", "crnn_vgg_s3",
    "abl_aug_none_s1", "abl_pub_48x368_s1", "abl_aug_heavy_s1",
    "crnn_convnext_s1", "svtr_small_s1", "diff_ocr_s1", "parseq_small_s1",
]

# Architectures whose params are fixed (some metrics.json have params_M=null).
PARAM_BACKFILL = {
    "abl_asp_48x512_s1": 8.9, "abl_asp_48x512_s2": 8.9, "abl_asp_48x512_s3": 8.9,
    "crnn_vgg_s1": 8.9, "crnn_vgg_s2": 8.9, "crnn_vgg_s3": 8.9,
    "abl_aug_none_s1": 8.9, "abl_pub_48x368_s1": 8.9, "abl_aug_heavy_s1": 8.9,
    "crnn_convnext_s1": 31.5, "svtr_small_s1": 3.9, "diff_ocr_s1": 10.0,
    "parseq_small_s1": 11.0,
}


def gather_ours() -> dict[str, dict]:
    rows = {}
    for p in sorted((PROJECT_ROOT / "results").glob("*/metrics.json")):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        t1 = m.get("test_I", {})
        t2 = m.get("test_II", {})
        rows[p.parent.name] = {
            "method": p.parent.name,
            "params_M": m.get("params_M") or PARAM_BACKFILL.get(p.parent.name),
            "T1_CER": t1.get("cer"), "T1_NED": t1.get("ned"),
            "T1_WER": t1.get("wer"), "T1_SeqAcc": t1.get("seq_acc"),
            "T2_CER": t2.get("cer"), "T2_NED": t2.get("ned"),
            "T2_WER": t2.get("wer"), "T2_SeqAcc": t2.get("seq_acc"),
            "ckpt": m.get("ckpt"),
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=PROJECT_ROOT / "tables", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ours = gather_ours()
    ordered = [ours[k] for k in ORDER if k in ours] + [v for k, v in ours.items() if k not in ORDER]

    # ---- CSV (raw internal names, unchanged) ----
    csv_path = args.out_dir / "main_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "method", "params_M", "T1_CER", "T1_NED", "T2_CER", "T2_NED", "T1_WER", "T2_WER", "T1_SeqAcc", "T2_SeqAcc"])
        for name, pM, c1, n1, c2, n2 in PUBLISHED:
            w.writerow(["Belay et al. 2024", name, pM, c1, n1, c2, n2, "", "", "", ""])
        for r in ordered:
            w.writerow(["ours", r["method"], r["params_M"], r["T1_CER"], r["T1_NED"], r["T2_CER"], r["T2_NED"], r["T1_WER"], r["T2_WER"], r["T1_SeqAcc"], r["T2_SeqAcc"]])
    print(f"wrote {csv_path}")

    # ---- mean +/- std for the aspect-preserved headline (3 seeds) ----
    asp = [ours[k] for k in ("abl_asp_48x512_s1", "abl_asp_48x512_s2", "abl_asp_48x512_s3") if k in ours]
    mean_row = None
    if len(asp) == 3:
        def mstd(key):
            vals = [r[key] for r in asp]
            return statistics.mean(vals), statistics.stdev(vals)  # sample std (n-1)
        t1c_m, t1c_s = mstd("T1_CER"); t1n_m, _ = mstd("T1_NED")
        t2c_m, t2c_s = mstd("T2_CER"); t2n_m, _ = mstd("T2_NED")
        mean_row = (t1c_m, t1c_s, t1n_m, t2c_m, t2c_s, t2n_m)

    # ---- LaTeX (publication table, readable names) ----
    def fmt(x):
        return "--" if x is None else f"{x:.2f}"

    tex = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{HHD-Ethiopic results (CER / NED, lower is better). Belay et al.\\ 2024 "
        "numbers are quoted verbatim from Table~3 of their paper; ``asp.'' denotes our "
        "aspect-preserved $48\\times512$ input, bare geometry the published $48\\times368$ "
        "hard-resize. The bold row is our headline three-seed mean.}",
        "\\label{tab:main-results}", "\\small",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Method & Params (M) & T1 CER & T1 NED & T2 CER & T2 NED \\\\", "\\midrule",
        "\\multicolumn{6}{l}{\\emph{Belay et al.\\ 2024 (published)}}\\\\",
    ]
    for name, pM, c1, n1, c2, n2 in PUBLISHED:
        pm = "--" if pM is None else f"{pM:.1f}"
        tex.append(f"\\quad {name} & {pm} & {c1:.2f} & {n1:.2f} & {c2:.2f} & {n2:.2f} \\\\")
    tex.append("\\midrule")
    tex.append("\\multicolumn{6}{l}{\\emph{Ours}}\\\\")
    if mean_row is not None:
        t1c_m, t1c_s, t1n_m, t2c_m, t2c_s, t2n_m = mean_row
        tex.append(
            f"\\quad \\textbf{{CRNN-VGG, $48\\times512$ asp.\\ (mean$\\pm$std, 3 seeds)}} & \\textbf{{8.9}} & "
            f"$\\mathbf{{{t1c_m:.2f}{{\\scriptstyle\\,\\pm\\,{t1c_s:.2f}}}}}$ & {t1n_m:.2f} & "
            f"$\\mathbf{{{t2c_m:.2f}{{\\scriptstyle\\,\\pm\\,{t2c_s:.2f}}}}}$ & {t2n_m:.2f} \\\\"
        )
    for r in ordered:
        disp = DISPLAY.get(r["method"], r["method"].replace("_", r"\_"))
        pm = "--" if r["params_M"] is None else f"{r['params_M']:.1f}"
        tex.append(f"\\quad {disp} & {pm} & {fmt(r['T1_CER'])} & {fmt(r['T1_NED'])} & {fmt(r['T2_CER'])} & {fmt(r['T2_NED'])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (args.out_dir / "main_results.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"wrote {args.out_dir/'main_results.tex'}")

    # ---- Markdown (raw internal names; for quick reading) ----
    md = ["# Main results: HHD-Ethiopic\n", "",
          "| Source | Method | Params (M) | T1 CER | T1 NED | T2 CER | T2 NED | T1 WER | T2 WER |",
          "|--------|--------|-----------:|-------:|-------:|-------:|-------:|-------:|-------:|"]
    for name, pM, c1, n1, c2, n2 in PUBLISHED:
        pm = "n/a" if pM is None else f"{pM:.1f}"
        md.append(f"| Belay 2024 | {name} | {pm} | {c1:.2f} | {n1:.2f} | {c2:.2f} | {n2:.2f} | n/a | n/a |")
    for r in ordered:
        pm = "n/a" if r["params_M"] is None else f"{r['params_M']:.1f}"
        md.append(f"| **ours** | {r['method']} | {pm} | {fmt(r['T1_CER'])} | {fmt(r['T1_NED'])} | {fmt(r['T2_CER'])} | {fmt(r['T2_NED'])} | {fmt(r['T1_WER'])} | {fmt(r['T2_WER'])} |")
    (args.out_dir / "main_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {args.out_dir/'main_results.md'}")


if __name__ == "__main__":
    main()
