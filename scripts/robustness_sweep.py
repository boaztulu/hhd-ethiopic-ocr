"""Run a trained checkpoint over corrupted versions of Test-I / Test-II.

Corruptions: gaussian blur, gaussian noise, brightness reduction. Emits a CSV +
plot of CER vs. corruption intensity.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import HHDImageDataset, ar_collate, ctc_collate
from src.data.transforms import base_transform, IMAGENET_MEAN, IMAGENET_STD
from src.data.vocab import CTCVocab, Vocab
from src.utils.metrics import compute_metrics


def make_corrupted_transform(h, w, kind: str, level: float, preserve_aspect: bool):
    """Add a corruption op before the standard normalize."""
    from albumentations.pytorch import ToTensorV2
    from src.data.transforms import _aspect_preserving_op
    import cv2

    if kind == "blur":
        k = max(3, int(round(level)) | 1)  # odd
        corrupt = A.GaussianBlur(blur_limit=(k, k), sigma_limit=(level / 2, level / 2), p=1.0)
    elif kind == "noise":
        corrupt = A.GaussNoise(var_limit=(level, level), mean=0, p=1.0)
    elif kind == "brightness":
        corrupt = A.RandomBrightnessContrast(brightness_limit=(-level, -level), contrast_limit=0, p=1.0)
    else:
        raise ValueError(kind)
    ops = []
    if preserve_aspect:
        ops.append(_aspect_preserving_op(h, w))
    else:
        ops.append(A.Resize(h, w))
    ops += [corrupt, A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def evaluate(model, loader, kind: str, vocab, device, max_len=80):
    preds, gts = [], []
    for batch in tqdm(loader, leave=False):
        images = batch["images"].to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            if kind == "ctc":
                logp = model(images)
                ids = logp.argmax(-1).t().cpu().numpy()
                for row in ids:
                    preds.append(vocab.decode_greedy(row.tolist()))
            else:
                ids = model.generate(images, max_len=max_len)
                if isinstance(ids, torch.Tensor):
                    ids = ids.cpu().tolist()
                for seq in ids:
                    preds.append(vocab.decode(seq))
        gts.extend(batch["texts"])
    return compute_metrics(preds, gts)


def build_model(ckpt):
    cfg = ckpt["config"]
    name = cfg["model"]["name"]
    kwargs = cfg["model"].get("kwargs", {})
    ctcvoc = CTCVocab.from_json()
    arvoc = Vocab.from_json()
    if name == "crnn":
        from src.models.crnn import CRNN, CRNNConfig
        return CRNN(CRNNConfig(n_classes=ctcvoc.n_classes, **kwargs)), "ctc", ctcvoc, cfg
    if name == "svtr":
        from src.models.svtr import SVTR, SVTRConfig
        return SVTR(SVTRConfig(n_classes=ctcvoc.n_classes, **kwargs)), "ctc", ctcvoc, cfg
    if name == "parseq":
        from src.models.parseq import PARSeq, PARSeqConfig
        return PARSeq(PARSeqConfig(n_classes=len(arvoc), sos_id=arvoc.sos_id, eos_id=arvoc.eos_id, pad_id=arvoc.pad_id, **kwargs)), "ar", arvoc, cfg
    if name == "trocr":
        from src.models.trocr_wrap import TrOCRWrap, TrOCRWrapConfig
        return TrOCRWrap(TrOCRWrapConfig(n_classes=len(arvoc), sos_id=arvoc.sos_id, eos_id=arvoc.eos_id, pad_id=arvoc.pad_id, **kwargs)), "ar", arvoc, cfg
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--split", default="test_II")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model, kind, vocab, cfg = build_model(ckpt)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device).eval()
    h, w = cfg["image"]["height"], cfg["image"]["width"]
    pres = cfg["image"].get("preserve_aspect", True)
    max_len = cfg.get("max_text_len", 80)

    grid = {
        "blur": [0, 1, 3, 5, 7, 9],
        "noise": [0, 5, 25, 75, 150, 300],
        "brightness": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }
    out_rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for kind_name, levels in grid.items():
        cers = []
        for lv in levels:
            if lv == 0 and kind_name != "noise":
                t = base_transform(h, w, preserve_aspect=pres)
            else:
                t = make_corrupted_transform(h, w, kind_name, lv, pres)
            ds = HHDImageDataset(args.split, vocab, transform=t, mode=kind, keep_text=True, max_text_len=max_len)
            collate = ctc_collate if kind == "ctc" else (lambda b: ar_collate(b, pad_id=vocab.pad_id))
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
            m = evaluate(model, loader, kind, vocab, args.device, max_len=max_len)
            cers.append(m.cer)
            out_rows.append({"corruption": kind_name, "level": lv, "cer": m.cer, "ned": m.ned, "wer": m.wer, "n": m.n})
            print(f"  {kind_name}={lv}  CER={m.cer:.3f}")
        ax.plot(levels, cers, marker="o", label=kind_name)
    ax.set_xlabel("corruption level")
    ax.set_ylabel("CER (%)")
    ax.set_title(f"Robustness ({args.split})  ckpt={args.ckpt.parent.name}")
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"robustness_{args.split}.{ext}", dpi=300)
    plt.close(fig)

    csv_path = args.out_dir / f"robustness_{args.split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["corruption", "level", "cer", "ned", "wer", "n"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
