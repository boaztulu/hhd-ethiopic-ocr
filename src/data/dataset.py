"""PyTorch Dataset for HHD-Ethiopic.

Two ingestion modes:
  * ``HHDImageDataset``: reads raw PNGs + CSV (slower but flexible).
  * ``HHDNumpyDataset``: loads the authors' pre-processed numpy arrays
    (``x_train.npy``, ``y_train.npy``): fast, fixed 48x368.

For CTC training, returns ``(image, target_ids, target_length)``.
For AR / encoder-decoder training, set ``mode='ar'`` to get sos/eos-padded ids.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .vocab import CTCVocab, Vocab

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "HHD-Ethiopic-data"))


SPLIT_PATHS = {
    "train": {
        "csv": DATA_ROOT / "train" / "train_raw" / "image_text_pairs_train.csv",
        "img_dir": DATA_ROOT / "train_raw" / "image_train",
        "x_npy": DATA_ROOT / "train" / "train_numpy" / "x_train.npy",
        "y_npy": DATA_ROOT / "train" / "train_numpy" / "y_train.npy",
    },
    "test_I": {
        "csv": DATA_ROOT / "test" / "test_rand" / "image_text_pairs_test_rand.csv",
        "img_dir": DATA_ROOT / "test" / "test_rand" / "image_rand",
    },
    "test_II": {
        "csv": DATA_ROOT / "test" / "test_18th" / "image_text_pairs_test_18th.csv",
        "img_dir": DATA_ROOT / "test" / "test_18th" / "image_18th",
    },
}


def _read_csv(path: Path) -> list[tuple[str, str]]:
    """HHD-Ethiopic CSVs have no header row: each line is `<filename>,<text>`.

    Ground-truth strings in the published CSV are right-padded with spaces to a
    fixed width; we strip them so target lengths reflect actual text length.
    """
    rows: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            text = row[1].rstrip(" ").rstrip()
            rows.append((row[0].strip(), text))
    return rows


class HHDImageDataset(Dataset):
    """Loads PNG + GT from the published raw layout."""

    def __init__(
        self,
        split: str,
        vocab: CTCVocab | Vocab,
        transform: Callable | None = None,
        mode: str = "ctc",  # 'ctc' or 'ar'
        max_text_len: int = 80,
        keep_text: bool = False,
    ) -> None:
        if split not in SPLIT_PATHS:
            raise ValueError(f"unknown split={split!r}; choose from {list(SPLIT_PATHS)}")
        paths = SPLIT_PATHS[split]
        if not paths["csv"].exists():
            raise FileNotFoundError(paths["csv"])
        self.split = split
        self.img_dir = paths["img_dir"]
        self.rows = _read_csv(paths["csv"])
        self.vocab = vocab
        self.transform = transform
        self.mode = mode
        self.max_text_len = max_text_len
        self.keep_text = keep_text

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        fname, text = self.rows[idx]
        path = self.img_dir / fname
        if not path.exists():
            alt = self.img_dir / Path(fname).name
            if alt.exists():
                path = alt
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        if self.transform is not None:
            tensor = self.transform(image=img)["image"]
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.mode == "ctc":
            assert isinstance(self.vocab, CTCVocab), "ctc mode requires CTCVocab"
            tgt = self.vocab.encode(text)[: self.max_text_len]
            out = {"image": tensor, "target": torch.tensor(tgt, dtype=torch.long), "target_length": len(tgt)}
        else:
            assert isinstance(self.vocab, Vocab), "ar mode requires Vocab"
            tgt = self.vocab.encode(text)[: self.max_text_len + 2]
            out = {"image": tensor, "target": torch.tensor(tgt, dtype=torch.long), "target_length": len(tgt)}

        if self.keep_text:
            out["text"] = text
            out["filename"] = fname
        return out


class HHDNumpyDataset(Dataset):
    """Fast path using authors' pre-processed numpy arrays.

    The arrays are (N, H, W) uint8 grayscale at 48x368 (CTC training input). We
    convert to 3-channel float and apply Normalize/ToTensor at __getitem__ time.
    """

    def __init__(
        self,
        x_path: str | Path,
        y_path: str | Path,
        vocab: CTCVocab,
        transform: Callable | None = None,
        max_text_len: int = 80,
        keep_text: bool = False,
        text_decoder: Callable[[np.ndarray], str] | None = None,
    ) -> None:
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r")
        self.vocab = vocab
        self.transform = transform
        self.max_text_len = max_text_len
        self.keep_text = keep_text
        self.text_decoder = text_decoder

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if x.ndim == 2:
            x = np.stack([x] * 3, axis=-1)
        # mmap arrays are read-only; copy so albumentations can mutate
        x = np.ascontiguousarray(x).astype(np.uint8)
        if self.transform is not None:
            tensor = self.transform(image=x)["image"]
        else:
            tensor = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0

        y_row = self.Y[idx]
        if self.text_decoder is not None:
            text = self.text_decoder(y_row)
            tgt = self.vocab.encode(text)[: self.max_text_len]
        else:
            # assume y_row is already a vector of label ids in author's vocab; fall back: drop zeros
            ids = [int(v) for v in y_row if int(v) > 0][: self.max_text_len]
            tgt = ids
            text = None
        out = {"image": tensor, "target": torch.tensor(tgt, dtype=torch.long), "target_length": len(tgt)}
        if self.keep_text and text is not None:
            out["text"] = text
        return out


def ctc_collate(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = torch.cat([b["target"] for b in batch], dim=0)
    target_lengths = torch.tensor([b["target_length"] for b in batch], dtype=torch.long)
    out = {"images": images, "targets": targets, "target_lengths": target_lengths}
    if "text" in batch[0]:
        out["texts"] = [b["text"] for b in batch]
        out["filenames"] = [b.get("filename") for b in batch]
    return out


def ar_collate(batch: list[dict], pad_id: int = 0) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    L = max(b["target_length"] for b in batch)
    targets = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        targets[i, : b["target_length"]] = b["target"]
    lengths = torch.tensor([b["target_length"] for b in batch], dtype=torch.long)
    out = {"images": images, "targets": targets, "target_lengths": lengths}
    if "text" in batch[0]:
        out["texts"] = [b["text"] for b in batch]
        out["filenames"] = [b.get("filename") for b in batch]
    return out
