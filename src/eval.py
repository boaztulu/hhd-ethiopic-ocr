"""Eval a trained checkpoint on val + Test-I + Test-II.

Outputs ``${OUT}/predictions_{split}.csv`` and ``${OUT}/metrics.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import HHDImageDataset, ar_collate, ctc_collate
from src.data.transforms import base_transform
from src.data.vocab import CTCVocab, Vocab
from src.utils.metrics import bootstrap_ci, compute_metrics


def build_model(ckpt: dict, n_ctc: int, n_ar: int):
    cfg = ckpt.get("config", {})
    name = cfg["model"]["name"]
    kwargs = cfg["model"].get("kwargs", {})
    if name == "crnn":
        from src.models.crnn import CRNN, CRNNConfig
        model = CRNN(CRNNConfig(n_classes=n_ctc, **kwargs))
        kind = "ctc"
    elif name == "svtr":
        from src.models.svtr import SVTR, SVTRConfig
        model = SVTR(SVTRConfig(n_classes=n_ctc, **kwargs))
        kind = "ctc"
    elif name == "parseq":
        from src.models.parseq import PARSeq, PARSeqConfig
        vocab = Vocab.from_json()
        model = PARSeq(PARSeqConfig(
            n_classes=n_ar, sos_id=vocab.sos_id, eos_id=vocab.eos_id, pad_id=vocab.pad_id, **kwargs,
        ))
        kind = "ar"
    elif name == "trocr":
        from src.models.trocr_wrap import TrOCRWrap, TrOCRWrapConfig
        vocab = Vocab.from_json()
        model = TrOCRWrap(TrOCRWrapConfig(
            n_classes=n_ar, sos_id=vocab.sos_id, eos_id=vocab.eos_id, pad_id=vocab.pad_id, **kwargs,
        ))
        kind = "ar"
    elif name == "diff_ocr":
        from src.models.diff_ocr import DiffOCR, DiffOCRConfig
        vocab = Vocab.from_json()
        diff = cfg.get("diffusion", {})
        model = DiffOCR(DiffOCRConfig(
            n_classes=len(vocab) + 1,
            mask_id=vocab.mask_id,
            pad_id=vocab.pad_id,
            eos_id=vocab.eos_id,
            max_len=diff.get("max_len", 50),
            img_size=(cfg["image"]["height"], cfg["image"]["width"]),
            embed_dim=kwargs.get("embed_dim", 256),
            dec_depth=kwargs.get("dec_depth", 6),
            num_heads=kwargs.get("num_heads", 4),
            n_diffusion_steps=diff.get("n_steps", 1000),
            inference_steps=diff.get("inference_steps", 12),
        ))
        kind = "diff"
    else:
        raise ValueError(name)
    model.load_state_dict(ckpt["model"], strict=False)
    return model, kind, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ctcvoc = CTCVocab.from_json()
    arvoc = Vocab.from_json()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model, kind, cfg = build_model(ckpt, ctcvoc.n_classes, len(arvoc))
    model.to(args.device).eval()
    vocab = ctcvoc if kind == "ctc" else arvoc

    h, w = cfg["image"]["height"], cfg["image"]["width"]
    preserve = cfg["image"].get("preserve_aspect", True)
    t = base_transform(h, w, preserve_aspect=preserve)

    all_metrics = {}
    for split in ["test_I", "test_II"]:
        # Diffusion model uses the AR-style dataset (full Vocab + sos/eos) for I/O
        ds_mode = "ar" if kind in ("ar", "diff") else "ctc"
        ds = HHDImageDataset(split, vocab, transform=t, mode=ds_mode, keep_text=True)
        collate = ctc_collate if kind == "ctc" else (lambda b: ar_collate(b, pad_id=arvoc.pad_id))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)

        preds, gts, fns = [], [], []
        for batch in tqdm(loader, desc=split, leave=False):
            images = batch["images"].to(args.device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                if kind == "ctc":
                    logp = model(images)  # (T, B, C)
                    ids = logp.argmax(-1).t().cpu().numpy()
                    for row in ids:
                        preds.append(ctcvoc.decode_greedy(row.tolist()))
                else:
                    if kind == "diff":
                        ids = model.generate(images)
                    else:
                        ids = model.generate(images, max_len=cfg["max_text_len"] + 2)
                    if isinstance(ids, torch.Tensor):
                        ids = ids.cpu().tolist()
                    for seq in ids:
                        preds.append(arvoc.decode(seq))
            gts.extend(batch["texts"])
            fns.extend(batch.get("filenames") or [None] * len(batch["texts"]))

        # Write predictions
        with open(args.output_dir / f"predictions_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w_ = csv.writer(f)
            w_.writerow(["filename", "ground_truth", "prediction"])
            for fn, g, p in zip(fns, gts, preds):
                w_.writerow([fn or "", g, p])

        m = compute_metrics(preds, gts)
        cer_lo, cer_hi = bootstrap_ci(preds, gts, "cer", n_boot=args.bootstrap)
        ned_lo, ned_hi = bootstrap_ci(preds, gts, "ned", n_boot=args.bootstrap)
        all_metrics[split] = {
            "cer": m.cer, "ned": m.ned, "wer": m.wer, "seq_acc": m.seq_acc, "n": m.n,
            "cer_ci95": [cer_lo, cer_hi], "ned_ci95": [ned_lo, ned_hi],
        }
        print(f"{split}  CER={m.cer:.3f}  NED={m.ned:.3f}  WER={m.wer:.3f}  SeqAcc={m.seq_acc:.3f}  CER95=[{cer_lo:.3f}, {cer_hi:.3f}]")

    all_metrics["ckpt"] = str(args.ckpt)
    all_metrics["model_name"] = cfg["model"]["name"]
    all_metrics["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)
    print(json.dumps(all_metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
