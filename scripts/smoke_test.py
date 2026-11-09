"""Smoke test: build vocab + dataset + model, run 2 fwd/bwd passes for CTC and AR.

Used to catch schema/shape/import errors before submitting multi-day SLURM jobs.
Runs on CPU if no GPU; takes <60s.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data.dataset import HHDImageDataset, ar_collate, ctc_collate
from src.data.transforms import base_transform, train_transform
from src.data.vocab import CTCVocab, Vocab
from src.models.crnn import CRNN, CRNNConfig
from src.models.parseq import PARSeq, PARSeqConfig
from src.models.svtr import SVTR, SVTRConfig


def main():
    print("Loading vocab...")
    ctcvoc = CTCVocab.from_json()
    arvoc = Vocab.from_json()
    print(f"  CTC classes: {ctcvoc.n_classes}  AR classes: {len(arvoc)}")

    h, w = 48, 368
    train_t = train_transform(h, w, aug_level="medium")
    val_t = base_transform(h, w)

    print("Loading 4 train items (CTC)...")
    ds = HHDImageDataset("train", ctcvoc, transform=train_t, mode="ctc", keep_text=True)
    print(f"  train size: {len(ds)}")
    batch = ctc_collate([ds[i] for i in range(4)])
    print(f"  images={tuple(batch['images'].shape)} targets_len={len(batch['targets'])} tgt_lens={batch['target_lengths'].tolist()} texts={batch['texts']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    print("CRNN-vgglike forward/backward...")
    model = CRNN(CRNNConfig(n_classes=ctcvoc.n_classes, backbone="vgglike")).to(device)
    images = batch["images"].to(device)
    logp = model(images)
    print(f"  output (T,B,C) = {tuple(logp.shape)}  n_params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    targets = batch["targets"].to(device)
    tlens = batch["target_lengths"].to(device)
    T = logp.size(0)
    ilens = torch.full((logp.size(1),), T, dtype=torch.long, device=device)
    loss = torch.nn.functional.ctc_loss(logp, targets, ilens, tlens, blank=0, zero_infinity=True)
    print(f"  loss = {loss.item():.3f}")
    loss.backward()
    print("  backward OK")
    del model

    print("SVTR forward...")
    model = SVTR(SVTRConfig(n_classes=ctcvoc.n_classes)).to(device)
    logp = model(images)
    print(f"  output (T,B,C) = {tuple(logp.shape)}  n_params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    del model

    print("AR datasets + PARSeq forward...")
    ar_ds = HHDImageDataset("train", arvoc, transform=val_t, mode="ar", keep_text=True)
    arbatch = ar_collate([ar_ds[i] for i in range(4)], pad_id=arvoc.pad_id)
    model = PARSeq(PARSeqConfig(n_classes=len(arvoc), sos_id=arvoc.sos_id, eos_id=arvoc.eos_id, pad_id=arvoc.pad_id)).to(device)
    images = arbatch["images"].to(device)
    targets = arbatch["targets"].to(device)
    logits = model(images, targets[:, :-1])
    print(f"  PARSeq logits = {tuple(logits.shape)}  n_params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    out_ids = model.generate(images, max_len=10)
    print(f"  generate ids shape = {tuple(out_ids.shape)}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
