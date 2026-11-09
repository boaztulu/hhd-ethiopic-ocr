"""CRNN with modern convolutional backbone, BiLSTM head, CTC loss.

Two backbone choices:
  * ``vgglike``:   replicates the classical CRNN (Shi et al. 2016) feature extractor.
  * ``convnext``:  ConvNeXt-Tiny features via timm, frozen-by-default first stage.

The input is (B, 3, H, W). With H=48 and a stride-4 height collapse, we get a
sequence of length W/4. We then run a 2-layer BiLSTM(256) and a per-step linear
to ``n_classes`` (= vocab.n_classes including blank).
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGLikeBackbone(nn.Module):
    """Classical CRNN feature extractor (Shi 2016)."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),  # 48 -> 24
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),  # 24 -> 12
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),  # 12 -> 6
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),  # 6 -> 3
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(inplace=True),  # 3 -> 2
        )
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns (B, C, H', W')
        return self.conv(x)


class ConvNeXtBackbone(nn.Module):
    """timm ConvNeXt-Tiny features, projected to C=512.

    We use timm's ``features_only=True`` to get the last stage feature map and
    then collapse the height to 1 via adaptive avg pool.
    """

    def __init__(self, model_name: str = "convnext_tiny", pretrained: bool = True, out_channels: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,),  # last stage
        )
        feat_chs = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv2d(feat_chs, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[-1]
        feats = self.proj(feats)
        return feats


@dataclass
class CRNNConfig:
    n_classes: int  # includes blank
    backbone: str = "convnext"  # 'convnext' or 'vgglike'
    lstm_hidden: int = 256
    lstm_layers: int = 2
    dropout: float = 0.1
    pretrained: bool = True


class CRNN(nn.Module):
    """CNN encoder → BiLSTM → linear → CTC."""

    def __init__(self, cfg: CRNNConfig):
        super().__init__()
        if cfg.backbone == "vgglike":
            self.backbone = VGGLikeBackbone()
        elif cfg.backbone == "convnext":
            self.backbone = ConvNeXtBackbone(pretrained=cfg.pretrained)
        else:
            raise ValueError(cfg.backbone)
        self.cfg = cfg
        self.rnn = nn.LSTM(
            input_size=self.backbone.out_channels,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            bidirectional=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(2 * cfg.lstm_hidden, cfg.n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> log-softmax over n_classes, shape (T, B, n_classes) for CTCLoss."""
        feats = self.backbone(images)  # (B, C, H', W')
        # Collapse H' by adaptive pool to 1 then squeeze
        if feats.size(2) > 1:
            feats = F.adaptive_avg_pool2d(feats, (1, feats.size(3)))
        feats = feats.squeeze(2)  # (B, C, W')
        feats = feats.permute(0, 2, 1)  # (B, W', C)
        rnn_out, _ = self.rnn(feats)
        logits = self.head(self.dropout(rnn_out))  # (B, T, n_classes)
        # CTCLoss wants (T, B, C) log-probs
        return logits.log_softmax(dim=-1).permute(1, 0, 2)

    @torch.no_grad()
    def predict_ids(self, images: torch.Tensor) -> torch.Tensor:
        logp = self.forward(images)  # (T, B, C)
        return logp.argmax(dim=-1).t()  # (B, T)
