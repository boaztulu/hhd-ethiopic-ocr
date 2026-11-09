"""Train DiffOCR: mask-based discrete diffusion recognizer.

Differences from ``train_ar.py``:
  * Vocab is ``len(vocab) + 1`` (extra MASK token).
  * Targets are right-padded with PAD to a fixed length ``cfg.diffusion.max_len``
    (default 50). The model predicts every position; loss is CE only on
    *masked* positions (with PAD positions allowed to be masked too; model
    should learn to predict PAD).
  * Validation / test uses ``model.generate()`` with iterative refinement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import HHDImageDataset, ar_collate
from src.data.transforms import base_transform, train_transform
from src.data.vocab import Vocab
from src.models.diff_ocr import DiffOCR, DiffOCRConfig
from src.utils.metrics import compute_metrics


SHOULD_STOP = False


def _on_sigusr1(signum, frame):  # pragma: no cover
    global SHOULD_STOP
    print("[SIGUSR1] received; saving and exiting.", flush=True)
    SHOULD_STOP = True


signal.signal(signal.SIGUSR1, _on_sigusr1)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def pad_to_max_len(ids: torch.Tensor, max_len: int, pad_id: int) -> torch.Tensor:
    """Pad a (L,) tensor to (max_len,) with pad_id."""
    out = torch.full((max_len,), pad_id, dtype=ids.dtype, device=ids.device)
    out[: min(len(ids), max_len)] = ids[: max_len]
    return out


def diff_collate(batch, pad_id: int, max_len: int):
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = torch.stack([pad_to_max_len(b["target"], max_len, pad_id) for b in batch], dim=0)
    lengths = torch.tensor([b["target_length"] for b in batch], dtype=torch.long)
    out = {"images": images, "targets": targets, "target_lengths": lengths}
    if "text" in batch[0]:
        out["texts"] = [b["text"] for b in batch]
        out["filenames"] = [b.get("filename") for b in batch]
    return out


def build_loaders(vocab: Vocab, cfg: dict):
    h, w = cfg["image"]["height"], cfg["image"]["width"]
    pres = cfg["image"].get("preserve_aspect", True)
    train_t = train_transform(h, w, aug_level=cfg["augment"]["level"], preserve_aspect=pres)
    val_t = base_transform(h, w, preserve_aspect=pres)
    train_ds = HHDImageDataset("train", vocab, transform=train_t, mode="ar", max_text_len=cfg["max_text_len"])
    test_I = HHDImageDataset("test_I", vocab, transform=val_t, mode="ar", max_text_len=cfg["max_text_len"], keep_text=True)
    test_II = HHDImageDataset("test_II", vocab, transform=val_t, mode="ar", max_text_len=cfg["max_text_len"], keep_text=True)
    rng = np.random.default_rng(cfg.get("val_seed", 0))
    n = len(train_ds); idx = np.arange(n); rng.shuffle(idx)
    n_val = int(round(0.10 * n)); val_idx, train_idx = idx[:n_val], idx[n_val:]
    val_ds = HHDImageDataset("train", vocab, transform=val_t, mode="ar", max_text_len=cfg["max_text_len"], keep_text=True)
    val_ds = Subset(val_ds, val_idx.tolist())
    train_ds = Subset(train_ds, train_idx.tolist())
    bs = cfg["train"]["batch_size"]; nw = cfg["train"].get("num_workers", 2)
    max_len = cfg["diffusion"]["max_len"]

    def collate(batch):
        return diff_collate(batch, pad_id=vocab.pad_id, max_len=max_len)

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=nw > 0),
        DataLoader(val_ds, batch_size=cfg["eval"].get("batch_size", bs * 2), shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=True, persistent_workers=nw > 0),
        DataLoader(test_I, batch_size=cfg["eval"].get("batch_size", bs * 2), shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=True, persistent_workers=nw > 0),
        DataLoader(test_II, batch_size=cfg["eval"].get("batch_size", bs * 2), shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=True, persistent_workers=nw > 0),
    )


@torch.no_grad()
def evaluate_diff(model: DiffOCR, loader, vocab: Vocab, device: str, max_batches: int | None = None) -> dict:
    model.eval()
    preds, gts = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches: break
        images = batch["images"].to(device, non_blocking=True)
        ids = model.generate(images)  # (B, L) with PAD-terminated sequences
        for row in ids.cpu().tolist():
            preds.append(vocab.decode(row))  # decode stops at EOS / pad
        gts.extend(batch["texts"])
    m = compute_metrics(preds, gts)
    return {"cer": m.cer, "ned": m.ned, "wer": m.wer, "seq_acc": m.seq_acc, "n": m.n}


def save_ckpt(path: Path, model, optim, sched, scaler, epoch, step, best, cfg):
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
    # n_classes = canonical vocab + 1 for the MASK token
    n_classes = len(vocab) + 1
    diff_cfg = DiffOCRConfig(
        n_classes=n_classes,
        mask_id=vocab.mask_id,  # = len(vocab)
        pad_id=vocab.pad_id,
        eos_id=vocab.eos_id,
        max_len=cfg["diffusion"]["max_len"],
        img_size=(cfg["image"]["height"], cfg["image"]["width"]),
        embed_dim=cfg["model"]["kwargs"].get("embed_dim", 256),
        dec_depth=cfg["model"]["kwargs"].get("dec_depth", 6),
        num_heads=cfg["model"]["kwargs"].get("num_heads", 4),
        n_diffusion_steps=cfg["diffusion"]["n_steps"],
        inference_steps=cfg["diffusion"]["inference_steps"],
    )
    model = DiffOCR(diff_cfg).to(args.device)
    print(f"model=DiffOCR vocab={n_classes}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    train_loader, val_loader, t1_loader, t2_loader = build_loaders(vocab, cfg)
    print(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test-I={len(t1_loader.dataset)}  test-II={len(t2_loader.dataset)}")

    epochs = cfg["train"]["epochs"]
    lr = float(cfg["train"]["lr"]); wd = float(cfg["train"].get("weight_decay", 0.05))
    warmup = int(cfg["train"].get("warmup_steps", 1000))
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    total_steps = epochs * (len(train_loader) // grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: min(1.0, s / max(1, warmup)) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps))))),
    )
    scaler = GradScaler("cuda")

    best = {"val_cer": float("inf"), "test_I_cer": None, "test_II_cer": None, "epoch": -1, "step": -1}
    start_epoch = 0; global_step = 0

    if args.init_from and args.init_from.exists():
        state = torch.load(args.init_from, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        try:
            optim.load_state_dict(state["optimizer"]);
            if state.get("scheduler"): sched.load_state_dict(state["scheduler"])
            if state.get("scaler"): scaler.load_state_dict(state["scaler"])
        except Exception as e:
            print(f"[resume] partial restore: {e}")
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("step", 0)
        best = state.get("best", best)
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    last_ckpt = args.output_dir / "last.pt"
    best_ckpt = args.output_dir / "best.pt"

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time(); running = 0.0; nb = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        optim.zero_grad(set_to_none=True)
        for it, batch in enumerate(pbar):
            images = batch["images"].to(args.device, non_blocking=True)
            targets = batch["targets"].to(args.device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                logits, mask, t = model(images, targets)
                # CE on masked positions only
                if mask.any():
                    loss = F.cross_entropy(logits[mask], targets[mask], label_smoothing=0.1)
                else:
                    loss = logits.sum() * 0.0  # rare; should never hit
            loss = loss / grad_accum
            scaler.scale(loss).backward()
            if (it + 1) % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True); sched.step()
                global_step += 1
            running += float(loss.item()) * grad_accum; nb += 1
            if it % 50 == 0:
                pbar.set_postfix(loss=f"{running/nb:.3f}", lr=f"{optim.param_groups[0]['lr']:.2e}")
            if SHOULD_STOP: break
        dt = time.time() - t0

        val_metrics = evaluate_diff(model, val_loader, vocab, args.device)
        row = {
            "epoch": epoch, "step": global_step, "train_loss": running / max(1, nb), "epoch_time_s": dt,
            "val_cer": val_metrics["cer"], "val_ned": val_metrics["ned"], "val_wer": val_metrics["wer"], "val_seq_acc": val_metrics["seq_acc"],
        }
        if epoch % 5 == 0 or epoch == epochs - 1:
            t1 = evaluate_diff(model, t1_loader, vocab, args.device)
            t2 = evaluate_diff(model, t2_loader, vocab, args.device)
            row.update({
                "test_I_cer": t1["cer"], "test_I_ned": t1["ned"], "test_I_wer": t1["wer"],
                "test_II_cer": t2["cer"], "test_II_ned": t2["ned"], "test_II_wer": t2["wer"],
            })
        else:
            t1 = t2 = None
        append_log(args.output_dir, row)
        print(json.dumps(row, default=float))

        save_ckpt(last_ckpt, model, optim, sched, scaler, epoch, global_step, best, cfg)
        if val_metrics["cer"] < best["val_cer"]:
            best.update({"val_cer": val_metrics["cer"], "epoch": epoch, "step": global_step,
                         "test_I_cer": (t1 or {}).get("cer"), "test_II_cer": (t2 or {}).get("cer")})
            save_ckpt(best_ckpt, model, optim, sched, scaler, epoch, global_step, best, cfg)
            print(f"[best] val_cer={best['val_cer']:.3f} @ epoch {epoch}")
        if SHOULD_STOP: break

    print(json.dumps({"final_best": best}, default=float))


if __name__ == "__main__":
    main()
