"""TrOCR fine-tune wrapper for HHD-Ethiopic.

We load HuggingFace TrOCR-small/base, swap its decoder vocab for the Ethiopic
character vocab (full Vocab including pad/sos/eos/unk), and expose a forward
+ generate API compatible with the rest of the codebase.

Reasoning for char-level vocab: TrOCR's default BPE was trained on Latin; for
Ethiopic, a fresh char-level vocab over our train chars is simpler and avoids
splitting Ethiopic syllabics into nonsense tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from transformers import VisionEncoderDecoderModel
    HAS_TROCR = True
except Exception:  # pragma: no cover
    HAS_TROCR = False


@dataclass
class TrOCRWrapConfig:
    n_classes: int
    sos_id: int
    eos_id: int
    pad_id: int
    base_name: str = "microsoft/trocr-small-handwritten"
    max_len: int = 64
    freeze_encoder_first_epoch: bool = True


class TrOCRWrap(nn.Module):
    def __init__(self, cfg: TrOCRWrapConfig):
        super().__init__()
        if not HAS_TROCR:
            raise RuntimeError("transformers TrOCR classes not available; pip install -U transformers")
        self.cfg = cfg
        self.model = VisionEncoderDecoderModel.from_pretrained(cfg.base_name)
        # Resize embeddings to our vocab size
        self.model.decoder.resize_token_embeddings(cfg.n_classes)
        self.model.config.decoder_start_token_id = cfg.sos_id
        self.model.config.pad_token_id = cfg.pad_id
        self.model.config.eos_token_id = cfg.eos_id
        self.model.config.vocab_size = cfg.n_classes
        self.model.decoder.config.vocab_size = cfg.n_classes
        self.model.generation_config.decoder_start_token_id = cfg.sos_id
        self.model.generation_config.pad_token_id = cfg.pad_id
        self.model.generation_config.eos_token_id = cfg.eos_id
        self.model.generation_config.bos_token_id = cfg.sos_id
        self.model.generation_config.max_length = cfg.max_len

    def freeze_encoder(self, freeze: bool) -> None:
        for p in self.model.encoder.parameters():
            p.requires_grad = not freeze

    def forward(self, images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TrOCR expects pixel_values in (B, 3, H, W). We assume images already preprocessed.
        out = self.model(pixel_values=images, labels=target)
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int | None = None, num_beams: int = 1):
        out = self.model.generate(
            pixel_values=images,
            max_length=max_len or self.cfg.max_len,
            num_beams=num_beams,
            early_stopping=True,
        )
        return out
