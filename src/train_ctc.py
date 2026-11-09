"""Training loop for CTC models (CRNN, SVTR).

Supports:
  * two-stage training (synthetic pretrain → real fine-tune) via two configs
  * AMP mixed precision
  * gradient accumulation
  * resume from checkpoint
  * SIGUSR1 graceful save (for SLURM preemption / requeue)
  * per-epoch CER on val + Test-I + Test-II
  * wandb (offline) and TensorBoard logging

Usage:
    python -m src.train_ctc --config configs/crnn_convnext.yaml \
        --output_dir checkpoints/crnn_convnext_real --seed 1
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import HHDImageDataset, ctc_collate
from src.data.transforms import base_transform, train_transform
from src.data.vocab import CTCVocab
from src.models.crnn import CRNN, CRNNConfig
from src.models.svtr import SVTR, SVTRConfig
from src.utils.metrics import compute_metrics


SHOULD_STOP = False


def _on_sigusr1(signum, frame):  # pragma: no cover
    global SHOULD_STOP
    print(f"\n[SIGUSR1] received; will save and exit gracefully.", flush=True)
    SHOULD_STOP = True


signal.signal(signal.SIGUSR1, _on_sigusr1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark stays True for speed; CTC kernels are nondeterministic
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_model(model_name: str, n_classes: int, model_kwargs: dict) -> nn.Module:
    if model_name == "crnn":
        return CRNN(CRNNConfig(n_classes=n_classes, **model_kwargs))
    if model_name == "svtr":
        return SVTR(SVTRConfig(n_classes=n_classes, **model_kwargs))
    raise ValueError(model_name)


def build_loaders(vocab: CTCVocab, cfg: dict):
    h, w = cfg["image"]["height"], cfg["image"]["width"]
    preserve = cfg["image"].get("preserve_aspect", True)
    train_t = train_transform(h, w, aug_level=cfg["augment"]["level"], preserve_aspect=preserve)
    val_t = base_transform(h, w, preserve_aspect=preserve)
    train_ds = HHDImageDataset("train", vocab, transform=train_t, mode="ctc", max_text_len=cfg["max_text_len"])
    test_I = HHDImageDataset("test_I", vocab, transform=val_t, mode="ctc", max_text_len=cfg["max_text_len"], keep_text=True)
    test_II = HHDImageDataset("test_II", vocab, transform=val_t, mode="ctc", max_text_len=cfg["max_text_len"], keep_text=True)
    # 10% val split (deterministic, seeded)
    rng = np.random.default_rng(cfg.get("val_seed", 0))
    n = len(train_ds)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(0.10 * n))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    val_ds = HHDImageDataset("train", vocab, transform=val_t, mode="ctc", max_text_len=cfg["max_text_len"], keep_text=True)
    val_ds = Subset(val_ds, val_idx.tolist())
    train_ds = Subset(train_ds, train_idx.tolist())

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=ctc_collate, pin_memory=True, drop_last=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg["eval"].get("batch_size", bs * 2), shuffle=False, num_workers=nw, collate_fn=ctc_collate, pin_memory=True, persistent_workers=nw > 0)
    t1_loader = DataLoader(test_I, batch_size=cfg["eval"].get("batch_size", bs * 2), shuffle=False, num_workers=nw, collate_fn=ctc_collate, pin_memory=True, persistent_workers=nw > 0)
    t2_loader = DataLoader(test_II, batch_size=cfg["eval"].get("batch_size", bs * 2), shuffle=False, num_workers=nw, collate_fn=ctc_collate, pin_memory=True, persistent_workers=nw > 0)
    return train_loader, val_loader, t1_loader, t2_loader


@torch.no_grad()
def evaluate_ctc(model: nn.Module, loader: DataLoader, vocab: CTCVocab, device: str, max_batches: int | None = None) -> dict:
    model.eval()
    preds_all: list[str] = []
    gts_all: list[str] = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["images"].to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logp = model(images)  # (T, B, C)
        ids = logp.argmax(-1).t().cpu().numpy()  # (B, T)
        for row in ids:
            preds_all.append(vocab.decode_greedy(row.tolist()))
        gts_all.extend(batch["texts"])
    m = compute_metrics(preds_all, gts_all)
    return {"cer": m.cer, "ned": m.ned, "wer": m.wer, "seq_acc": m.seq_acc, "n": m.n, "preds": preds_all, "gts": gts_all}


def save_checkpoint(path: Path, model: nn.Module, optim, sched, scaler, epoch: int, step: int, best: dict, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best": best,
        "config": cfg,
    }, path)


def load_checkpoint(path: Path, model: nn.Module, optim=None, sched=None, scaler=None, device="cuda") -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    if optim is not None and "optimizer" in state:
        try:
            optim.load_state_dict(state["optimizer"])
        except Exception as e:
            print(f"[resume] optimizer state could not be restored: {e}")
    if sched is not None and state.get("scheduler") is not None:
        try:
            sched.load_state_dict(state["scheduler"])
        except Exception as e:
            print(f"[resume] scheduler state could not be restored: {e}")
    if scaler is not None and state.get("scaler") is not None:
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception as e:
            print(f"[resume] scaler state could not be restored: {e}")
    return state


def append_log(out_dir: Path, row: dict) -> None:
    p = out_dir / "metrics.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--init_from", type=Path, default=None, help="warm-start weights, ignore optimizer state")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_every_epoch", action="store_true", default=True)
    ap.add_argument("--no_eval_test", action="store_true", help="skip test-I/II eval during epochs (just val)")
    ap.add_argument("--max_steps", type=int, default=None, help="override epochs by step budget")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    seed_everything(args.seed)

    vocab = CTCVocab.from_json()
    n_classes = vocab.n_classes

    model = build_model(cfg["model"]["name"], n_classes, cfg["model"].get("kwargs", {}))
    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={cfg['model']['name']}  n_classes={n_classes}  params={n_params/1e6:.2f}M")

    train_loader, val_loader, t1_loader, t2_loader = build_loaders(vocab, cfg)
    print(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test-I={len(t1_loader.dataset)}  test-II={len(t2_loader.dataset)}")

    epochs = cfg["train"]["epochs"]
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.01))
    warmup_steps = int(cfg["train"].get("warmup_steps", 500))
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    total_steps = epochs * (len(train_loader) // grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda s: min(1.0, s / max(1, warmup_steps)) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps))))),
    )
    scaler = GradScaler("cuda")
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")

    best = {"val_cer": float("inf"), "test_I_cer": None, "test_II_cer": None, "epoch": -1, "step": -1}
    start_epoch = 0
    global_step = 0

    if args.init_from is not None and args.init_from.exists():
        print(f"warm-start from {args.init_from}")
        state = torch.load(args.init_from, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
    if args.resume is not None and args.resume.exists():
        print(f"resume from {args.resume}")
        state = load_checkpoint(args.resume, model, optim, sched, scaler, args.device)
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("step", 0)
        best = state.get("best", best)

    last_ckpt = args.output_dir / "last.pt"
    best_ckpt = args.output_dir / "best.pt"

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        optim.zero_grad(set_to_none=True)
        for it, batch in enumerate(pbar):
            images = batch["images"].to(args.device, non_blocking=True)
            targets = batch["targets"].to(args.device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(args.device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                logp = model(images)  # (T, B, C)
                T = logp.size(0)
                input_lengths = torch.full((logp.size(1),), T, dtype=torch.long, device=args.device)
                loss = loss_fn(logp, targets, input_lengths, target_lengths)
            loss = loss / grad_accum
            scaler.scale(loss).backward()
            if (it + 1) % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1
            running += float(loss.item()) * grad_accum
            n_batches += 1
            if it % 50 == 0:
                pbar.set_postfix(loss=f"{running / n_batches:.3f}", lr=f"{optim.param_groups[0]['lr']:.2e}")
            if SHOULD_STOP or (args.max_steps and global_step >= args.max_steps):
                break

        train_loss = running / max(1, n_batches)
        dt = time.time() - t0

        val_metrics = evaluate_ctc(model, val_loader, vocab, args.device)
        row = {
            "epoch": epoch, "step": global_step, "train_loss": train_loss, "epoch_time_s": dt,
            "val_cer": val_metrics["cer"], "val_ned": val_metrics["ned"], "val_wer": val_metrics["wer"], "val_seq_acc": val_metrics["seq_acc"],
        }
        if not args.no_eval_test:
            t1_metrics = evaluate_ctc(model, t1_loader, vocab, args.device)
            t2_metrics = evaluate_ctc(model, t2_loader, vocab, args.device)
            row.update({
                "test_I_cer": t1_metrics["cer"], "test_I_ned": t1_metrics["ned"], "test_I_wer": t1_metrics["wer"], "test_I_seq_acc": t1_metrics["seq_acc"],
                "test_II_cer": t2_metrics["cer"], "test_II_ned": t2_metrics["ned"], "test_II_wer": t2_metrics["wer"], "test_II_seq_acc": t2_metrics["seq_acc"],
            })
        else:
            t1_metrics = t2_metrics = None
        append_log(args.output_dir, row)
        print(json.dumps({k: v for k, v in row.items() if k != "preds"}, default=float))

        # checkpoint
        save_checkpoint(last_ckpt, model, optim, sched, scaler, epoch, global_step, best, cfg)
        if val_metrics["cer"] < best["val_cer"]:
            best.update({
                "val_cer": val_metrics["cer"], "epoch": epoch, "step": global_step,
                "test_I_cer": (t1_metrics or {}).get("cer"),
                "test_II_cer": (t2_metrics or {}).get("cer"),
            })
            save_checkpoint(best_ckpt, model, optim, sched, scaler, epoch, global_step, best, cfg)
            print(f"[best] new best val_cer={best['val_cer']:.3f} @ epoch {epoch}")

        if SHOULD_STOP:
            print("Graceful stop on SIGUSR1.")
            break
        if args.max_steps and global_step >= args.max_steps:
            print("Max steps reached.")
            break

    print(json.dumps({"final_best": best}, default=float))


if __name__ == "__main__":
    main()
