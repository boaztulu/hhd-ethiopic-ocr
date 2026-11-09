"""Training loop for AR / encoder-decoder models (PARSeq, TrOCR).

This is structurally similar to ``train_ctc.py`` but uses the full ``Vocab``
(with sos/eos/pad/unk) and cross-entropy loss on shifted targets. Generation at
eval time uses model.generate().
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import HHDImageDataset, ar_collate
from src.data.transforms import base_transform, train_transform
from src.data.vocab import Vocab
from src.utils.metrics import compute_metrics


SHOULD_STOP = False


def _on_sigusr1(signum, frame):  # pragma: no cover
    global SHOULD_STOP
    print("[SIGUSR1] received; saving and exiting.", flush=True)
    SHOULD_STOP = True


signal.signal(signal.SIGUSR1, _on_sigusr1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_model(model_name: str, vocab: Vocab, model_kwargs: dict) -> nn.Module:
    if model_name == "parseq":
        from src.models.parseq import PARSeq, PARSeqConfig
        cfg = PARSeqConfig(
            n_classes=len(vocab), sos_id=vocab.sos_id, eos_id=vocab.eos_id, pad_id=vocab.pad_id,
            **model_kwargs,
        )
        return PARSeq(cfg)
    if model_name == "trocr":
        from src.models.trocr_wrap import TrOCRWrap, TrOCRWrapConfig
        cfg = TrOCRWrapConfig(
            n_classes=len(vocab), sos_id=vocab.sos_id, eos_id=vocab.eos_id, pad_id=vocab.pad_id,
            **model_kwargs,
        )
        return TrOCRWrap(cfg)
    raise ValueError(model_name)


def build_loaders(vocab: Vocab, cfg: dict):
    h, w = cfg["image"]["height"], cfg["image"]["width"]
    preserve = cfg["image"].get("preserve_aspect", True)
    train_t = train_transform(h, w, aug_level=cfg["augment"]["level"], preserve_aspect=preserve)
    val_t = base_transform(h, w, preserve_aspect=preserve)
    train_ds = HHDImageDataset("train", vocab, transform=train_t, mode="ar", max_text_len=cfg["max_text_len"])
    test_I = HHDImageDataset("test_I", vocab, transform=val_t, mode="ar", max_text_len=cfg["max_text_len"], keep_text=True)
    test_II = HHDImageDataset("test_II", vocab, transform=val_t, mode="ar", max_text_len=cfg["max_text_len"], keep_text=True)
    rng = np.random.default_rng(cfg.get("val_seed", 0))
    n = len(train_ds)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(0.10 * n))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    val_ds = HHDImageDataset("train", vocab, transform=val_t, mode="ar", max_text_len=cfg["max_text_len"], keep_text=True)
    val_ds = Subset(val_ds, val_idx.tolist())
    train_ds = Subset(train_ds, train_idx.tolist())
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"].get("num_workers", 4)

    def collate(batch):
        return ar_collate(batch, pad_id=vocab.pad_id)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg["eval"].get("batch_size", bs * 2), shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=True, persistent_workers=nw > 0)
    t1_loader = DataLoader(test_I, batch_size=cfg["eval"].get("batch_size", bs * 2), shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=True, persistent_workers=nw > 0)
    t2_loader = DataLoader(test_II, batch_size=cfg["eval"].get("batch_size", bs * 2), shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=True, persistent_workers=nw > 0)
    return train_loader, val_loader, t1_loader, t2_loader


def make_targets_for_loss(targets: torch.Tensor, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift targets: input is [sos, ...], label is [..., eos].

    Returns (decoder_input_ids, labels).
    """
    dec_in = targets[:, :-1].contiguous()
    labels = targets[:, 1:].contiguous().clone()
    labels[labels == pad_id] = -100
    return dec_in, labels


@torch.no_grad()
def evaluate_ar(model, loader, vocab: Vocab, device: str, max_batches: int | None = None, num_beams: int = 1) -> dict:
    model.eval()
    preds_all, gts_all = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["images"].to(device, non_blocking=True)
        if hasattr(model, "generate"):
            ids = model.generate(images, max_len=loader.dataset.dataset.max_text_len if hasattr(loader.dataset, "dataset") else 80)
            if isinstance(ids, torch.Tensor):
                ids = ids.cpu().tolist()
            for seq in ids:
                preds_all.append(vocab.decode(seq))
        gts_all.extend(batch["texts"])
    m = compute_metrics(preds_all, gts_all)
    return {"cer": m.cer, "ned": m.ned, "wer": m.wer, "seq_acc": m.seq_acc, "n": m.n}


def save_checkpoint(path: Path, model, optim, sched, scaler, epoch, step, best, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "step": step, "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best": best, "config": cfg,
    }, path)


def append_log(out_dir: Path, row: dict) -> None:
    with open(out_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--init_from", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    seed_everything(args.seed)

    vocab = Vocab.from_json()
    model = build_model(cfg["model"]["name"], vocab, cfg["model"].get("kwargs", {}))
    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={cfg['model']['name']}  vocab={len(vocab)}  params={n_params/1e6:.2f}M")

    train_loader, val_loader, t1_loader, t2_loader = build_loaders(vocab, cfg)
    print(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test-I={len(t1_loader.dataset)}  test-II={len(t2_loader.dataset)}")

    epochs = cfg["train"]["epochs"]
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"].get("weight_decay", 0.01))
    warmup = int(cfg["train"].get("warmup_steps", 500))
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    total_steps = epochs * (len(train_loader) // grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: min(1.0, s / max(1, warmup)) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps))))),
    )
    scaler = GradScaler("cuda")

    best = {"val_cer": float("inf"), "test_I_cer": None, "test_II_cer": None, "epoch": -1, "step": -1}
    start_epoch = 0
    global_step = 0

    if args.init_from and args.init_from.exists():
        state = torch.load(args.init_from, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        print(f"warm-started from {args.init_from}")
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optim.load_state_dict(state["optimizer"])
        if state.get("scheduler"): sched.load_state_dict(state["scheduler"])
        if state.get("scaler"): scaler.load_state_dict(state["scaler"])
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("step", 0)
        best = state.get("best", best)
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    last_ckpt = args.output_dir / "last.pt"
    best_ckpt = args.output_dir / "best.pt"

    name = cfg["model"]["name"]
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        optim.zero_grad(set_to_none=True)
        for it, batch in enumerate(pbar):
            images = batch["images"].to(args.device, non_blocking=True)
            targets = batch["targets"].to(args.device, non_blocking=True)
            if name == "trocr":
                # TrOCR handles its own shifting; pass full ids with pad masked to -100
                labels = targets.clone()
                labels[labels == vocab.pad_id] = -100
                with autocast(device_type="cuda", dtype=torch.float16):
                    loss, _ = model(images, labels)
            else:
                dec_in, labels = make_targets_for_loss(targets, vocab.pad_id)
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images, dec_in)
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100,
                        label_smoothing=0.1,
                    )
            loss = loss / grad_accum
            scaler.scale(loss).backward()
            if (it + 1) % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1
            running += float(loss.item()) * grad_accum
            nb += 1
            if it % 50 == 0:
                pbar.set_postfix(loss=f"{running/nb:.3f}", lr=f"{optim.param_groups[0]['lr']:.2e}")
            if SHOULD_STOP:
                break
        dt = time.time() - t0

        val_metrics = evaluate_ar(model, val_loader, vocab, args.device)
        row = {
            "epoch": epoch, "step": global_step, "train_loss": running / max(1, nb), "epoch_time_s": dt,
            "val_cer": val_metrics["cer"], "val_ned": val_metrics["ned"], "val_wer": val_metrics["wer"], "val_seq_acc": val_metrics["seq_acc"],
        }
        # Skip Test eval most epochs (AR generation is slow): every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            t1_metrics = evaluate_ar(model, t1_loader, vocab, args.device)
            t2_metrics = evaluate_ar(model, t2_loader, vocab, args.device)
            row.update({
                "test_I_cer": t1_metrics["cer"], "test_I_ned": t1_metrics["ned"], "test_I_wer": t1_metrics["wer"],
                "test_II_cer": t2_metrics["cer"], "test_II_ned": t2_metrics["ned"], "test_II_wer": t2_metrics["wer"],
            })
        else:
            t1_metrics = t2_metrics = None
        append_log(args.output_dir, row)
        print(json.dumps(row, default=float))

        save_checkpoint(last_ckpt, model, optim, sched, scaler, epoch, global_step, best, cfg)
        if val_metrics["cer"] < best["val_cer"]:
            best.update({
                "val_cer": val_metrics["cer"], "epoch": epoch, "step": global_step,
                "test_I_cer": (t1_metrics or {}).get("cer"),
                "test_II_cer": (t2_metrics or {}).get("cer"),
            })
            save_checkpoint(best_ckpt, model, optim, sched, scaler, epoch, global_step, best, cfg)
            print(f"[best] val_cer={best['val_cer']:.3f} @ epoch {epoch}")
        if SHOULD_STOP:
            break

    print(json.dumps({"final_best": best}, default=float))


if __name__ == "__main__":
    main()
