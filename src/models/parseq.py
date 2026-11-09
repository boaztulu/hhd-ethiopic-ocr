"""Minimal PARSeq-style permuted-autoregressive decoder for Ethiopic.

This is a pragmatic re-implementation: encoder = patch-embedded ViT-tiny over
the line image; decoder = causal transformer that predicts the next character
given an arbitrary permutation of positions, exposed during training. At
inference we use left-to-right autoregressive decoding.

We deliberately keep this small (the full PARSeq codebase is large) and rely on
the same encoder backbone shape as SVTR for fair compute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class EncoderViT(nn.Module):
    def __init__(self, embed_dim: int = 256, depth: int = 6, num_heads: int = 4, img_size=(48, 512), max_tokens: int = 4096):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1), nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 2, 1), nn.GELU(),  # /8 in H, /8 in W
        )
        # Oversize pos embedding so we don't crash on wider inputs.
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.patch_embed(images)  # (B, C, H', W')
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = tokens + self.pos[:, : tokens.size(1)]
        tokens = self.encoder(tokens)
        return self.norm(tokens)


@dataclass
class PARSeqConfig:
    n_classes: int  # includes sos/eos/pad/unk (i.e. full Vocab)
    sos_id: int
    eos_id: int
    pad_id: int
    max_len: int = 50
    embed_dim: int = 256
    enc_depth: int = 6
    dec_depth: int = 4
    num_heads: int = 4
    perm_train: bool = True
    n_perms: int = 6  # how many random permutations per batch


class PARSeq(nn.Module):
    def __init__(self, cfg: PARSeqConfig, img_size=(48, 512)):
        super().__init__()
        self.cfg = cfg
        self.encoder = EncoderViT(embed_dim=cfg.embed_dim, depth=cfg.enc_depth, num_heads=cfg.num_heads, img_size=img_size)
        self.tok_embed = nn.Embedding(cfg.n_classes, cfg.embed_dim)
        self.pos_embed = PositionalEncoding(cfg.embed_dim, max_len=cfg.max_len + 2)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dim, nhead=cfg.num_heads, dim_feedforward=cfg.embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.dec_depth)
        self.head = nn.Linear(cfg.embed_dim, cfg.n_classes)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def _perm_mask(self, perm: torch.Tensor) -> torch.Tensor:
        T = perm.numel()
        # For permuted AR: position i attends to all j s.t. rank(j) < rank(i)
        rank = torch.argsort(perm)
        m = torch.full((T, T), float("-inf"), device=perm.device)
        for i in range(T):
            ok = rank < rank[i]
            m[i, ok] = 0.0
        m.fill_diagonal_(0.0)
        return m

    def forward(self, images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(images)
        B, T = target.shape
        tok = self.tok_embed(target)
        tok = self.pos_embed(tok)
        mask = self._causal_mask(T, images.device)
        out = self.decoder(tok, memory, tgt_mask=mask)
        logits = self.head(out)
        return logits

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        memory = self.encoder(images)
        B = images.size(0)
        L = max_len or self.cfg.max_len
        ids = torch.full((B, 1), self.cfg.sos_id, dtype=torch.long, device=images.device)
        for _ in range(L):
            tok = self.tok_embed(ids)
            tok = self.pos_embed(tok)
            mask = self._causal_mask(tok.size(1), images.device)
            out = self.decoder(tok, memory, tgt_mask=mask)
            logits = self.head(out[:, -1])
            next_id = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if (next_id == self.cfg.eos_id).all():
                break
        return ids[:, 1:]  # drop sos
