"""Inference helper for HHD-Ethiopic checkpoints.

Provides:
  * ``HHDRecognizer``: a high-level class that loads a checkpoint and exposes
    ``predict(image_path | PIL.Image | np.ndarray)`` returning the Ethiopic
    transcription as a string.
  * Multi-architecture aware: detects CRNN / SVTR / PARSeq / DiffOCR from the
    checkpoint config, builds the right model and decoder.
  * A small CLI: ``python -m src.infer --ckpt CKPT --image PATH [--image PATH...]``.

Designed to be safe to import from a Gradio app (no SLURM-specific paths, no
side effects on import).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import torch
from PIL import Image

# Make ``src.*`` importable when this file is invoked directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.transforms import base_transform
from src.data.vocab import CTCVocab, Vocab

ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]


def _open_image(x: ImageLike) -> np.ndarray:
    """Normalise to a HxWx3 uint8 numpy RGB array."""
    if isinstance(x, np.ndarray):
        img = x
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.shape[-1] == 4:  # drop alpha
            img = img[..., :3]
        return img
    if isinstance(x, Image.Image):
        return np.array(x.convert("RGB"))
    p = Path(x)
    if not p.exists():
        raise FileNotFoundError(p)
    return np.array(Image.open(p).convert("RGB"))


def _build_model(ckpt: dict, ctcvoc: CTCVocab, arvoc: Vocab):
    cfg = ckpt.get("config", {})
    name = cfg["model"]["name"]
    kwargs = cfg["model"].get("kwargs", {})
    if name == "crnn":
        from src.models.crnn import CRNN, CRNNConfig
        model = CRNN(CRNNConfig(n_classes=ctcvoc.n_classes, **kwargs))
        kind = "ctc"
    elif name == "svtr":
        from src.models.svtr import SVTR, SVTRConfig
        model = SVTR(SVTRConfig(n_classes=ctcvoc.n_classes, **kwargs))
        kind = "ctc"
    elif name == "parseq":
        from src.models.parseq import PARSeq, PARSeqConfig
        model = PARSeq(PARSeqConfig(
            n_classes=len(arvoc), sos_id=arvoc.sos_id, eos_id=arvoc.eos_id,
            pad_id=arvoc.pad_id, **kwargs,
        ))
        kind = "ar"
    elif name == "diff_ocr":
        from src.models.diff_ocr import DiffOCR, DiffOCRConfig
        diff = cfg.get("diffusion", {})
        model = DiffOCR(DiffOCRConfig(
            n_classes=len(arvoc) + 1,
            mask_id=arvoc.mask_id,
            pad_id=arvoc.pad_id,
            eos_id=arvoc.eos_id,
            max_len=diff.get("max_len", 50),
            img_size=(cfg["image"]["height"], cfg["image"]["width"]),
            embed_dim=kwargs.get("embed_dim", 256),
            dec_depth=kwargs.get("dec_depth", 6),
            num_heads=kwargs.get("num_heads", 4),
            n_diffusion_steps=diff.get("n_steps", 1000),
            inference_steps=diff.get("inference_steps", 12),
        ))
        kind = "diff"
    elif name == "trocr":
        from src.models.trocr_wrap import TrOCRWrap, TrOCRWrapConfig
        model = TrOCRWrap(TrOCRWrapConfig(
            n_classes=len(arvoc), sos_id=arvoc.sos_id, eos_id=arvoc.eos_id,
            pad_id=arvoc.pad_id, **kwargs,
        ))
        kind = "ar"
    else:
        raise ValueError(f"unknown model name in ckpt: {name!r}")
    model.load_state_dict(ckpt["model"], strict=False)
    return model, kind, cfg


class HHDRecognizer:
    """Load an HHD-Ethiopic checkpoint and recognise lines.

    Example
    -------
    >>> rec = HHDRecognizer("checkpoints/abl_asp_48x512_s2/best.pt")
    >>> rec.predict("HHD-Ethiopic-data/test/test_rand/image_rand/test_rand_00000.png")
    'ፅ፡ዳዊት፡፻፵፱፡ሰብሕዎ፡'
    """

    def __init__(self, ckpt_path: Union[str, os.PathLike], device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.ckpt_path = Path(ckpt_path)
        ctcvoc = CTCVocab.from_json()
        arvoc = Vocab.from_json()
        ckpt = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        model, kind, cfg = _build_model(ckpt, ctcvoc, arvoc)
        self.model = model.to(device).eval()
        self.kind = kind
        self.cfg = cfg
        self.ctcvoc = ctcvoc
        self.arvoc = arvoc
        self.transform = base_transform(
            cfg["image"]["height"],
            cfg["image"]["width"],
            preserve_aspect=cfg["image"].get("preserve_aspect", True),
        )
        self.model_name = cfg["model"]["name"]
        self.n_params_M = sum(p.numel() for p in self.model.parameters()) / 1e6

    # -------- public API --------

    @torch.no_grad()
    def predict(self, image: ImageLike) -> str:
        """Recognise a single image. Returns the predicted Ethiopic string."""
        return self.predict_batch([image])[0]

    @torch.no_grad()
    def predict_batch(self, images: Iterable[ImageLike], batch_size: int = 32) -> list[str]:
        """Recognise a list of images. Returns a list of predicted strings."""
        items = list(images)
        out: list[str] = []
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            tensors = [self._to_tensor(_open_image(x)) for x in chunk]
            batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
            if self.kind == "ctc":
                logp = self.model(batch)  # (T, B, C)
                ids = logp.argmax(-1).t().cpu().tolist()
                out.extend(self.ctcvoc.decode_greedy(row) for row in ids)
            elif self.kind == "diff":
                ids = self.model.generate(batch).cpu().tolist()
                out.extend(self.arvoc.decode(row) for row in ids)
            else:  # ar
                max_len = self.cfg.get("max_text_len", 60) + 2
                ids = self.model.generate(batch, max_len=max_len)
                ids = ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids
                out.extend(self.arvoc.decode(row) for row in ids)
        return out

    # -------- helpers --------

    def _to_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        return self.transform(image=img_rgb)["image"]

    def info(self) -> dict:
        return {
            "ckpt": str(self.ckpt_path),
            "model": self.model_name,
            "params_M": round(self.n_params_M, 2),
            "input_hw": [self.cfg["image"]["height"], self.cfg["image"]["width"]],
            "preserve_aspect": self.cfg["image"].get("preserve_aspect", True),
            "device": self.device,
        }


# ---------- CLI ----------

def _list_checkpoints(root: Path) -> list[Path]:
    return sorted(p for p in (root / "checkpoints").glob("*/best.pt"))


def main():
    ap = argparse.ArgumentParser(description="HHD-Ethiopic single-line recogniser.")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Path to checkpoint best.pt. If omitted, lists available checkpoints.")
    ap.add_argument("--image", action="append", default=[], type=Path,
                    help="Image path(s) to recognise. May be repeated.")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (autodetect by default)")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    if args.ckpt is None:
        print("Available checkpoints:")
        for p in _list_checkpoints(ROOT):
            print(f"  {p.relative_to(ROOT)}")
        return

    rec = HHDRecognizer(args.ckpt, device=args.device)
    print("Recognizer loaded:", rec.info())
    if not args.image:
        print("Pass --image PATH (may be repeated) to recognise images.")
        return
    preds = rec.predict_batch(args.image, batch_size=args.batch_size)
    for p, t in zip(args.image, preds):
        print(f"{p}: {t}")


if __name__ == "__main__":
    main()
