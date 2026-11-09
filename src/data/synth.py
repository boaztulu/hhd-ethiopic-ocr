"""Authors-provided synthetic supplement for underrepresented characters.

The HHD-Ethiopic release ships 1,200 synthetic text-line images under
``train_raw/under_represented_char_synth/``: each is a binarised PNG with a
matching ``.gt.txt`` ground truth. We mix these in with the real train data to
help the model see rare characters more often.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from .vocab import CTCVocab, Vocab

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYNTH_DIR = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "HHD-Ethiopic-data")) / "train_raw" / "under_represented_char_synth"


class SynthSupplementDataset(Dataset):
    """Authors' synthetic supplement for rare Ethiopic characters."""

    def __init__(
        self,
        synth_dir: str | Path = DEFAULT_SYNTH_DIR,
        vocab: CTCVocab | Vocab | None = None,
        transform: Callable | None = None,
        mode: str = "ctc",
        max_text_len: int = 80,
    ):
        self.synth_dir = Path(synth_dir)
        if not self.synth_dir.exists():
            raise FileNotFoundError(self.synth_dir)
        self.pngs = sorted(self.synth_dir.glob("*.bin.png"))
        if not self.pngs:
            raise RuntimeError(f"no *.bin.png under {self.synth_dir}")
        self.vocab = vocab
        self.transform = transform
        self.mode = mode
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.pngs)

    def __getitem__(self, idx: int):
        img_path = self.pngs[idx]
        gt_path = img_path.with_name(img_path.stem.replace(".bin", "") + ".gt.txt")
        if not gt_path.exists():
            # Fall back: 010000.bin.png → 010000.gt.txt
            stem = img_path.name.replace(".bin.png", "")
            gt_path = img_path.parent / f"{stem}.gt.txt"
        text = gt_path.read_text(encoding="utf-8").strip()

        im = Image.open(img_path).convert("RGB")
        x = np.array(im)
        if self.transform is not None:
            tensor = self.transform(image=x)["image"]
        else:
            tensor = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0

        if self.mode == "ctc":
            assert isinstance(self.vocab, CTCVocab)
            tgt = self.vocab.encode(text)[: self.max_text_len]
        else:
            assert isinstance(self.vocab, Vocab)
            tgt = self.vocab.encode(text)[: self.max_text_len + 2]
        return {"image": tensor, "target": torch.tensor(tgt, dtype=torch.long), "target_length": len(tgt), "text": text}


def mix_with_real(real_ds, synth_ds, synth_repeat: int = 1):
    """Concatenate synth repeated `synth_repeat` times to the real dataset."""
    return ConcatDataset([real_ds] + [synth_ds] * synth_repeat)
