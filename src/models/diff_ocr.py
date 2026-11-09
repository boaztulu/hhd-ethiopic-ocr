"""DiffOCR: a discrete mask-based diffusion recognizer for Ethiopic.

Design (absorbing-state discrete diffusion, equivalent to MaskGIT for text):
  * Visual encoder produces (B, T_v, C) features from the line image.
  * A learned MASK token + the vocab tokens populate an output token sequence of
    fixed length T_y aligned to the visual sequence.
  * A small transformer decoder with cross-attention to visual features takes
    the (possibly masked) token sequence and the diffusion step t, outputs
    per-position logits over the vocabulary.
  * Training: randomly mask tokens at ratio ``m ~ Uniform[0, 1]``, replacing
    with ``<MASK>``. The decoder predicts ALL positions; loss is CE on the
    masked positions only (cosine-schedule weighting optional).
  * Inference (12-step iterative refinement; see ``inference_steps``):
      x_0 = [MASK, MASK, ..., MASK]
      for t in T..1:
          logits = model(x_t, visual, t)
          predict tokens; keep top-(1 - m_t) most confident, re-mask the rest
      x_final = argmax(logits)

This is the recipe used in IDM-Diff (Tang 2023, Zheng 2024 for sequence labelling).
Reference: Austin et al. 2021 "Structured Denoising Diffusion Models in Discrete State-Spaces".

We use a tiny SVTR-derived encoder for visual features (same as SVTR-Small) so
this is genuinely a *diffusion* recognizer, distinct from the AR / CTC models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .svtr import (
    Block,
    HeightMerge,
    PatchEmbed,
)


# ---------- Encoder (re-uses SVTR-Small blocks for parameter efficiency) ----------

class DiffEncoder(nn.Module):
    """Lightweight SVTR-style visual encoder.

    Outputs (B, T_v, C) where T_v = image_width / 4 (= 128 for 48x512).
    """

    def __init__(self, img_size=(48, 512), embed_dims=(96, 192, 256), depths=(3, 4, 3), num_heads=(3, 6, 8)):
        super().__init__()
        self.img_size = img_size
        self.patch = PatchEmbed(3, embed_dims[0])
        self.stage1 = nn.ModuleList([Block(embed_dims[0], num_heads[0], window=(7, 11)) for _ in range(depths[0])])
        self.down1 = HeightMerge(embed_dims[0], embed_dims[1])
        self.stage2 = nn.ModuleList([Block(embed_dims[1], num_heads[1], window=(7, 11)) for _ in range(depths[1])])
        self.down2 = HeightMerge(embed_dims[1], embed_dims[2])
        self.stage3 = nn.ModuleList([Block(embed_dims[2], num_heads[2], window=None) for _ in range(depths[2])])
        self.norm = nn.LayerNorm(embed_dims[2])
        self.out_dim = embed_dims[2]

    def _run(self, blocks, x_bchw):
        B, C, H, W = x_bchw.shape
        x = x_bchw.flatten(2).transpose(1, 2)
        for blk in blocks:
            x = blk(x, H, W)
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch(images)
        x = self._run(self.stage1, x)
        x = self.down1(x)
        x = self._run(self.stage2, x)
        x = self.down2(x)
        x = self._run(self.stage3, x)
        # Collapse height to 1, return (B, T_v, C)
        x = F.adaptive_avg_pool2d(x, (1, x.size(3))).squeeze(2)  # (B, C, T_v)
        x = x.permute(0, 2, 1)
        return self.norm(x)


# ---------- Decoder ----------

class StepEmbed(nn.Module):
    """Sinusoidal time-step embedding + MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return self.mlp(emb)


class DiffDecoder(nn.Module):
    """Transformer decoder over the token sequence with cross-attention to visual features.

    Self-attention is BIDIRECTIONAL (no causal mask); diffusion decoders see the
    entire current state and refine in parallel.
    """

    def __init__(self, n_classes: int, max_len: int, embed_dim: int, depth: int, num_heads: int):
        super().__init__()
        self.tok = nn.Embedding(n_classes, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.step = StepEmbed(embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x_t: torch.Tensor, memory: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: (B, L) token ids; memory: (B, T_v, C); t: (B,)
        emb = self.tok(x_t) + self.pos[:, : x_t.size(1)]
        emb = emb + self.step(t).unsqueeze(1)  # broadcast over positions
        # No causal mask: all positions attend to all
        out = self.decoder(emb, memory)
        return self.head(self.norm(out))  # (B, L, n_classes)


# ---------- Full model ----------

@dataclass
class DiffOCRConfig:
    n_classes: int        # full vocab including MASK
    mask_id: int          # token id of MASK
    pad_id: int           # token id of PAD (used to fill unused positions)
    eos_id: int           # token id of EOS (used as a length terminator)
    max_len: int = 50     # max number of token positions
    img_size: tuple[int, int] = (48, 512)
    embed_dim: int = 256
    dec_depth: int = 6
    num_heads: int = 4
    n_diffusion_steps: int = 1000  # for the step embedding range
    inference_steps: int = 12      # how many refine iterations at sampling time (runs used 12)


class DiffOCR(nn.Module):
    def __init__(self, cfg: DiffOCRConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = DiffEncoder(img_size=cfg.img_size, embed_dims=(96, 192, cfg.embed_dim))
        self.decoder = DiffDecoder(
            n_classes=cfg.n_classes,
            max_len=cfg.max_len,
            embed_dim=cfg.embed_dim,
            depth=cfg.dec_depth,
            num_heads=cfg.num_heads,
        )

    # ---- Training step ----

    def mask_inputs(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask tokens with a sampled ratio per item.

        Crucially we mask **all** positions including PAD; the model must learn
        to predict PAD at "no character here" positions so that, at inference,
        starting from all-MASK it can shape the right output length.

        Returns (x_t, mask, t) where mask is True at masked positions and t is
        the integer step index (used for the step embedding).
        """
        B, L = target.shape
        device = target.device
        # Sample mask ratio per item from Uniform[0, 1] (MaskGIT-style)
        ratio = torch.rand(B, device=device)
        # Convert ratio to "step": t = floor(ratio * T)
        t = (ratio * (self.cfg.n_diffusion_steps - 1)).long()

        rand = torch.rand(B, L, device=device)
        mask = rand < ratio.unsqueeze(-1)
        # Ensure at least one masked position per item so loss is non-degenerate
        no_mask_items = ~mask.any(dim=-1)
        if no_mask_items.any():
            # Force one mask at position 0 for those items
            mask[no_mask_items, 0] = True

        x_t = target.clone()
        x_t[mask] = self.cfg.mask_id
        return x_t, mask, t

    def forward(self, images: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward.

        Returns (logits, mask, target); the caller computes CE on masked positions.
        """
        # Build memory from visual features
        memory = self.encoder(images)  # (B, T_v, C)
        x_t, mask, t = self.mask_inputs(target)
        logits = self.decoder(x_t, memory, t)
        return logits, mask, target

    # ---- Inference ----

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        """Iterative-refinement sampling (MaskGIT cosine schedule).

        At step i (1-indexed, of T total) we leave ``L * cos(i/T * π/2)`` positions
        masked. The positions kept masked are the lowest-confidence currently-masked
        positions. The rest get committed to their top-1 prediction.

        Returns predicted token ids (B, L).
        """
        L = max_len or self.cfg.max_len
        B = images.size(0)
        device = images.device
        memory = self.encoder(images)

        x = torch.full((B, L), self.cfg.mask_id, dtype=torch.long, device=device)
        T = self.cfg.inference_steps

        for step_idx in range(T):
            t_emb = torch.full(
                (B,),
                int((1 - step_idx / max(1, T)) * (self.cfg.n_diffusion_steps - 1)),
                dtype=torch.long,
                device=device,
            )
            logits = self.decoder(x, memory, t_emb)  # (B, L, C)
            probs = logits.softmax(-1)
            top_probs, top_ids = probs.max(-1)

            if step_idx == T - 1:
                # Final step: commit everything still masked.
                x = torch.where(x == self.cfg.mask_id, top_ids, x)
                break

            # How many positions to keep masked after this step.
            frac_next = math.cos((step_idx + 1) / T * math.pi / 2)
            n_keep_masked = max(0, int(L * frac_next))

            cur_mask = (x == self.cfg.mask_id)
            # Score "keep me masked" = -confidence for currently masked positions,
            # -inf for non-masked positions (they should NEVER be re-masked).
            score = torch.where(cur_mask, -top_probs, torch.full_like(top_probs, -float("inf")))
            if n_keep_masked > 0:
                _, keep_idx = torch.topk(score, k=n_keep_masked, dim=-1)
                still_masked = torch.zeros_like(cur_mask)
                still_masked.scatter_(-1, keep_idx, True)
            else:
                still_masked = torch.zeros_like(cur_mask)
            # Positions to commit at this step: currently masked AND not in still_masked.
            commit = cur_mask & ~still_masked
            x = torch.where(commit, top_ids, x)

        # Final safety: ensure no MASK remains.
        if (x == self.cfg.mask_id).any():
            t_emb = torch.zeros(B, dtype=torch.long, device=device)
            logits = self.decoder(x, memory, t_emb)
            x = torch.where(x == self.cfg.mask_id, logits.argmax(-1), x)
        return x
