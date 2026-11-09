"""SVTR (single visual model) text recognizer adapted to HHD-Ethiopic.

This is a minimal, faithful re-implementation of the SVTR design (Du et al.
2022, IJCAI). It uses:
  * patch embedding with 4× height reduction
  * 3 stages of mixed local-window / global self-attention blocks
  * 1-D combining (height collapse) + projection to ``n_classes`` (CTC head)

We keep the implementation small (<300 LOC, single file) so that we control
preprocessing and tokenisation end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None, act: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PatchEmbed(nn.Module):
    """4x downsample patch embed (two 3x3 stride-2 convs)."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            ConvBNAct(in_ch, embed_dim // 2, k=3, s=2),
            ConvBNAct(embed_dim // 2, embed_dim, k=3, s=2),
        )

    def forward(self, x):  # (B, 3, H, W) -> (B, C, H/4, W/4)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MSA(nn.Module):
    """Multi-head self-attention, optional local window over a fixed (H, W) grid."""

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0, window: tuple[int, int] | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.window = window

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.window is None:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).reshape(B, N, C)
        else:
            wh, ww = self.window
            # reshape to (B*Nh, Nw_h, Nw_w, heads, hd)
            assert N == H * W
            q = q.transpose(1, 2).reshape(B, H, W, self.num_heads, self.head_dim)
            k = k.transpose(1, 2).reshape(B, H, W, self.num_heads, self.head_dim)
            v = v.transpose(1, 2).reshape(B, H, W, self.num_heads, self.head_dim)
            ph = (wh - H % wh) % wh
            pw = (ww - W % ww) % ww
            if ph or pw:
                q = F.pad(q, (0, 0, 0, 0, 0, pw, 0, ph))
                k = F.pad(k, (0, 0, 0, 0, 0, pw, 0, ph))
                v = F.pad(v, (0, 0, 0, 0, 0, pw, 0, ph))
            Hp, Wp = H + ph, W + pw
            q = q.view(B, Hp // wh, wh, Wp // ww, ww, self.num_heads, self.head_dim).permute(0, 1, 3, 5, 2, 4, 6).contiguous()
            k = k.view(B, Hp // wh, wh, Wp // ww, ww, self.num_heads, self.head_dim).permute(0, 1, 3, 5, 2, 4, 6).contiguous()
            v = v.view(B, Hp // wh, wh, Wp // ww, ww, self.num_heads, self.head_dim).permute(0, 1, 3, 5, 2, 4, 6).contiguous()
            q = q.view(-1, self.num_heads, wh * ww, self.head_dim)
            k = k.view(-1, self.num_heads, wh * ww, self.head_dim)
            v = v.view(-1, self.num_heads, wh * ww, self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
            out = out.view(B, Hp // wh, Wp // ww, self.num_heads, wh, ww, self.head_dim).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
            out = out.reshape(B, Hp, Wp, C)
            if ph or pw:
                out = out[:, :H, :W, :].contiguous()
            out = out.reshape(B, N, C)
        return self.drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, window: tuple[int, int] | None = None, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MSA(dim, num_heads, drop=drop, window=window)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class HeightMerge(nn.Module):
    """Downsample height by 2 via 1x3 conv stride 2 in H."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))


@dataclass
class SVTRConfig:
    n_classes: int
    img_size: tuple[int, int] = (48, 368)  # H, W
    embed_dims: tuple[int, int, int] = (96, 192, 256)
    depths: tuple[int, int, int] = (3, 6, 3)
    num_heads: tuple[int, int, int] = (3, 6, 8)
    windows: tuple[tuple[int, int] | None, ...] = ((7, 11), (7, 11), None)  # last stage = global
    mlp_ratio: float = 2.0
    drop: float = 0.1


class SVTR(nn.Module):
    def __init__(self, cfg: SVTRConfig):
        super().__init__()
        self.cfg = cfg
        H0, W0 = cfg.img_size
        self.patch_embed = PatchEmbed(3, cfg.embed_dims[0])
        # After patch embed: H/4, W/4
        self.h = [H0 // 4, H0 // 8, H0 // 16]
        self.w = [W0 // 4, W0 // 4, W0 // 4]

        self.stage1 = nn.ModuleList([
            Block(cfg.embed_dims[0], cfg.num_heads[0], window=cfg.windows[0], mlp_ratio=cfg.mlp_ratio, drop=cfg.drop)
            for _ in range(cfg.depths[0])
        ])
        self.down1 = HeightMerge(cfg.embed_dims[0], cfg.embed_dims[1])
        self.stage2 = nn.ModuleList([
            Block(cfg.embed_dims[1], cfg.num_heads[1], window=cfg.windows[1], mlp_ratio=cfg.mlp_ratio, drop=cfg.drop)
            for _ in range(cfg.depths[1])
        ])
        self.down2 = HeightMerge(cfg.embed_dims[1], cfg.embed_dims[2])
        self.stage3 = nn.ModuleList([
            Block(cfg.embed_dims[2], cfg.num_heads[2], window=cfg.windows[2], mlp_ratio=cfg.mlp_ratio, drop=cfg.drop)
            for _ in range(cfg.depths[2])
        ])
        # Combine: collapse remaining H to 1 by avg pool, then linear to n_classes
        self.norm = nn.LayerNorm(cfg.embed_dims[2])
        self.head = nn.Linear(cfg.embed_dims[2], cfg.n_classes)

    def _run_stage(self, blocks, x_bchw):
        B, C, H, W = x_bchw.shape
        x = x_bchw.flatten(2).transpose(1, 2)  # (B, N, C)
        for blk in blocks:
            x = blk(x, H, W)
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        x = self._run_stage(self.stage1, x)
        x = self.down1(x)
        x = self._run_stage(self.stage2, x)
        x = self.down2(x)
        x = self._run_stage(self.stage3, x)
        x = F.adaptive_avg_pool2d(x, (1, x.size(3))).squeeze(2)  # (B, C, W')
        x = x.permute(0, 2, 1)  # (B, W', C)
        x = self.norm(x)
        logits = self.head(x)  # (B, T, n_classes)
        return logits.log_softmax(-1).permute(1, 0, 2)  # (T, B, n_classes)
