"""Character vocabulary for Ethiopic OCR.

Loads `cache/vocab.json` produced by `scripts/phase1_eda.py`.

The vocabulary uses 5 special tokens then the train characters:
    0 <pad>   1 <sos>   2 <eos>   3 <unk>   4 <blank>   5..N+4 chars

For **CTC** training, we use a *separate* CTC vocab with the blank at index 0
(idiomatic for ``torch.nn.CTCLoss(blank=0)``) and the visible characters
shifted to 1..N; this is what ``CTCVocab`` exposes.

For **autoregressive** decoders we use the full vocab including pad/sos/eos/unk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "cache" / "vocab.json"


@dataclass
class Vocab:
    """Full vocabulary (special tokens + chars).

    Layout: 0=pad, 1=sos, 2=eos, 3=unk, 4=blank, 5..N+4 = chars.
    DiffOCR appends one extra MASK token at index ``len(vocab)``, not part of
    the canonical vocab so existing PARSeq / TrOCR checkpoints remain valid.
    """

    chars: list[str]
    pad_id: int = 0
    sos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    blank_id: int = 4

    @property
    def mask_id(self) -> int:
        return len(self)  # one past the canonical end; only used by DiffOCR


    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_VOCAB_PATH) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        chars: list[str] = d["chars"]
        return cls(chars=chars)

    @property
    def n_special(self) -> int:
        return 5

    @property
    def n_chars(self) -> int:
        return len(self.chars)

    def __len__(self) -> int:
        return self.n_special + self.n_chars

    @cached_property
    def char_to_id(self) -> dict[str, int]:
        return {c: i + self.n_special for i, c in enumerate(self.chars)}

    @cached_property
    def id_to_char(self) -> dict[int, str]:
        return {i + self.n_special: c for i, c in enumerate(self.chars)}

    def encode(self, text: str, add_sos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.char_to_id.get(c, self.unk_id) for c in text]
        if add_sos:
            ids = [self.sos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.sos_id):
                continue
            if i == self.unk_id:
                out.append("□")  # ▢ placeholder
            else:
                out.append(self.id_to_char.get(int(i), ""))
        return "".join(out)


@dataclass
class CTCVocab:
    """CTC-style vocab: blank at index 0, chars 1..N. UNK collapses to blank.

    This matches the convention of ``torch.nn.CTCLoss(blank=0)``.
    """

    chars: list[str]
    blank_id: int = 0

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_VOCAB_PATH) -> "CTCVocab":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(chars=list(d["chars"]))

    @property
    def n_classes(self) -> int:
        return 1 + len(self.chars)  # blank + chars

    def __len__(self) -> int:
        return self.n_classes

    @cached_property
    def char_to_id(self) -> dict[str, int]:
        # 1..N
        return {c: i + 1 for i, c in enumerate(self.chars)}

    @cached_property
    def id_to_char(self) -> dict[int, str]:
        return {i + 1: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        ids = []
        for c in text:
            ids.append(self.char_to_id.get(c, self.blank_id))
        # Strip any blank stand-ins (unk -> blank); CTC needs no blank in target
        return [i for i in ids if i != self.blank_id]

    def decode_greedy(self, ids: Sequence[int]) -> str:
        """Collapse repeats + remove blanks (CTC best-path)."""
        out: list[str] = []
        prev = -1
        for i in ids:
            i = int(i)
            if i != prev and i != self.blank_id:
                if i in self.id_to_char:
                    out.append(self.id_to_char[i])
            prev = i
        return "".join(out)
