"""Computational cost of each recogniser: parameters, multiply-accumulates, latency.

Answers the practical question of what the discrete-diffusion recogniser costs
relative to the convolutional-recurrent one.  DiffOCR runs its visual encoder
once and its decoder ``inference_steps`` times per image (see
``DiffOCR.generate``), so its inference cost is structurally different from a
single-pass CTC model even at comparable parameter counts.

Parameter counts and MACs are exact and hardware-independent.  Latency is
measured on whatever device is available and the device is reported alongside;
the trained checkpoints are loaded so the timing reflects the real models.

Usage:
    python scripts/compute_cost.py                # params + MACs only (fast)
    python scripts/compute_cost.py --latency      # adds the timing benchmark

Writes ``tables/compute_cost.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.vocab import CTCVocab, Vocab  # noqa: E402

# (label, checkpoint dir, input height, input width)
MODELS = [
    ("CRNN-VGG", "abl_asp_48x512_s2", 48, 512),
    ("CRNN-ConvNeXt", "crnn_convnext_s1", 48, 512),
    ("SVTR-Small", "svtr_small_s1", 48, 512),
    ("DiffOCR", "diff_ocr_s1", 48, 512),
]


def build(ckpt_path: Path):
    """Rebuild a model exactly as src/eval.py does, and load its trained weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    name = cfg["model"]["name"]
    kwargs = cfg["model"].get("kwargs", {})
    n_ctc = CTCVocab.from_json().n_classes
    vocab = Vocab.from_json()

    if name == "crnn":
        from src.models.crnn import CRNN, CRNNConfig
        model = CRNN(CRNNConfig(n_classes=n_ctc, **kwargs))
    elif name == "svtr":
        from src.models.svtr import SVTR, SVTRConfig
        model = SVTR(SVTRConfig(n_classes=n_ctc, **kwargs))
    elif name == "diff_ocr":
        from src.models.diff_ocr import DiffOCR, DiffOCRConfig
        diff = cfg.get("diffusion", {})
        model = DiffOCR(DiffOCRConfig(
            n_classes=len(vocab) + 1,
            mask_id=vocab.mask_id,
            pad_id=vocab.pad_id,
            eos_id=vocab.eos_id,
            max_len=diff.get("max_len", 50),
            n_diffusion_steps=diff.get("n_steps", 1000),
            inference_steps=diff.get("inference_steps", 12),
            **kwargs,
        ))
    else:
        raise ValueError(name)

    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, name, cfg


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def macs_of(module, *inputs) -> float | None:
    """MACs for one call of `module`, via thop. Returns None if it cannot trace.

    CAVEAT -- do not compare these across architecture families.  thop counts
    Conv2d, Linear and LSTM correctly (its LSTM figure was checked against an
    analytic count and agrees to 0.5%), but it has no rule for
    ``nn.MultiheadAttention`` and returns zero for it, and it never counts the
    raw ``q @ k.T`` / ``attn @ v`` matmuls in hand-written attention.  The MAC
    column is therefore complete for the CRNN models and an undercount for SVTR
    and for the DiffOCR decoder.  The measured latency below is the figure to
    quote when comparing designs.
    """
    try:
        from thop import profile
        import copy
        macs, _ = profile(copy.deepcopy(module), inputs=inputs, verbose=False)
        return float(macs)
    except Exception as exc:  # pragma: no cover - reported, not silently dropped
        print(f"    [thop failed: {type(exc).__name__}: {exc}]")
        return None


@torch.no_grad()
def time_model(model, name, h, w, batch, threads, warmup=5, iters=30):
    """Milliseconds per image: (median, min, stdev) over `iters` timed repetitions.

    The median and the minimum are reported because a wall-clock benchmark on a
    shared machine is contaminated upward by whatever else is running; the
    minimum over many repetitions is the best available estimate of the
    uncontended cost, and the median is robust to occasional stalls.  The mean
    is not reported for that reason.
    """
    torch.set_num_threads(threads)
    x = torch.zeros(batch, 3, h, w)

    def run():
        if name == "diff_ocr":
            return model.generate(x)
        return model.predict_ids(x) if hasattr(model, "predict_ids") else model(x)

    for _ in range(warmup):
        run()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1000.0)
    per_image = sorted(t / batch for t in times)
    return (statistics.median(per_image), per_image[0],
            statistics.stdev(per_image) if len(per_image) > 1 else 0.0)


def training_wallclock(run: str):
    """Total logged training time and epoch count from the per-epoch log."""
    p = Path("checkpoints") / run / "metrics.jsonl"
    if not p.exists():
        return None, None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    secs = sum(r.get("epoch_time_s", 0.0) for r in rows)
    return secs, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", action="store_true", help="run the timing benchmark")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--out", default=Path("tables/compute_cost.csv"), type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"platform: {platform.processor() or platform.machine()} | "
          f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | "
          f"threads: {args.threads}")

    rows = []
    for label, run, h, w in MODELS:
        ck = Path("checkpoints") / run / "best.pt"
        if not ck.exists():
            print(f"{label}: checkpoint missing ({ck}) - skipped")
            continue
        model, name, cfg = build(ck)
        n_par = count_params(model)
        secs, n_ep = training_wallclock(run)

        x1 = torch.zeros(1, 3, h, w)
        rec = {
            "model": label, "run": run, "params": n_par, "params_M": round(n_par / 1e6, 3),
            "input": f"{h}x{w}",
            "train_seconds": round(secs, 1) if secs else "",
            "train_epochs": n_ep or "",
            "train_hours": round(secs / 3600, 2) if secs else "",
        }

        if name == "diff_ocr":
            enc = macs_of(model.encoder, x1)
            mem = model.encoder(x1)
            tok = torch.full((1, model.cfg.max_len), model.cfg.mask_id, dtype=torch.long)
            temb = torch.zeros(1, dtype=torch.long)
            dec = macs_of(model.decoder, tok, mem, temb)
            steps = model.cfg.inference_steps
            rec["encoder_MMACs"] = round(enc / 1e6, 1) if enc else ""
            rec["decoder_MMACs_per_step"] = round(dec / 1e6, 1) if dec else ""
            rec["decoder_steps"] = steps
            rec["total_MMACs"] = round((enc + dec * steps) / 1e6, 1) if (enc and dec) else ""
            print(f"{label}: params={n_par:,} enc={enc/1e6 if enc else 0:.1f} MMAC "
                  f"dec={dec/1e6 if dec else 0:.1f} MMAC x {steps} steps "
                  f"-> total {(enc + dec*steps)/1e6 if enc and dec else 0:.1f} MMAC")
        else:
            tot = macs_of(model, x1)
            rec["encoder_MMACs"] = ""
            rec["decoder_MMACs_per_step"] = ""
            rec["decoder_steps"] = 1
            rec["total_MMACs"] = round(tot / 1e6, 1) if tot else ""
            print(f"{label}: params={n_par:,} total={tot/1e6 if tot else 0:.1f} MMAC (single pass)")

        if secs:
            print(f"    training: {secs/3600:.2f} h over {n_ep} epochs "
                  f"({secs/max(1, n_ep):.1f} s/epoch)")

        if args.latency:
            for b in args.batches:
                med, lo, sd = time_model(model, name, h, w, b, args.threads)
                rec[f"ms_median_bs{b}"] = round(med, 2)
                rec[f"ms_min_bs{b}"] = round(lo, 2)
                rec[f"ms_sd_bs{b}"] = round(sd, 2)
                print(f"    latency bs={b}: median {med:.2f} ms/image, min {lo:.2f} ms/image "
                      f"(sd {sd:.2f}; {1000/med:.1f} img/s at the median)")
        rows.append(rec)

    if rows:
        keys = sorted({k for r in rows for k in r})
        order = ["model", "run", "params", "params_M", "input", "decoder_steps",
                 "train_epochs", "train_seconds", "train_hours",
                 "total_MMACs", "encoder_MMACs", "decoder_MMACs_per_step"]
        keys = order + [k for k in keys if k not in order]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
