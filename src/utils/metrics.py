"""Recognition metrics that match the published HHD-Ethiopic definitions.

CER (Eq. 3 in Belay et al. 2024):
    CER = (1/c) * sum_i ED(pred_i, gt_i) * 100
    where c = total chars in *ground truth*.
    Note: this can exceed 100 when predictions are much longer than GT.

NED (Eq. 4):
    NED = (1/N) * sum_i ED(pred_i, gt_i) / max(|pred_i|, |gt_i|) * 100

WER (Ethiopic-aware word segmentation; not reported in the paper, we publish it):
    Words are split on the Ethiopic punctuation block U+1360..U+1368 plus ASCII
    whitespace. Standard: sum(ED words) / sum(ref words) * 100.

Sequence accuracy:
    Fraction of test items where pred == gt, as a percent.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import editdistance
import numpy as np

# Ethiopic word separators: the full Unicode punctuation block U+1360..U+1368
# plus regular whitespace. The most common are U+1361 (፡, Ethiopic wordspace)
# and U+1362 (።, full stop). The character class ፠-፨ covers the whole
# block (including U+1367, the Ethiopic question mark) so it matches the paper.
ETHIOPIC_SEP = re.compile(r"[\s፠-፨]+")


def _ethiopic_words(s: str) -> list[str]:
    return [w for w in ETHIOPIC_SEP.split(s) if w]


@dataclass
class RecMetrics:
    cer: float
    ned: float
    wer: float
    seq_acc: float
    n: int


def _ed(a: str, b: str) -> int:
    return int(editdistance.eval(a, b))


def compute_metrics(preds: Iterable[str], gts: Iterable[str]) -> RecMetrics:
    preds = list(preds)
    gts = list(gts)
    assert len(preds) == len(gts), f"len mismatch: {len(preds)} vs {len(gts)}"
    if not preds:
        return RecMetrics(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    total_gt_chars = sum(len(g) for g in gts)
    total_char_ed = 0
    ned_sum = 0.0
    seq_correct = 0

    total_word_ed = 0
    total_ref_words = 0

    for p, g in zip(preds, gts):
        ce = _ed(p, g)
        total_char_ed += ce
        denom = max(len(p), len(g), 1)
        ned_sum += ce / denom
        if p == g:
            seq_correct += 1
        wp = _ethiopic_words(p)
        wg = _ethiopic_words(g)
        total_word_ed += int(editdistance.eval(wp, wg))
        total_ref_words += len(wg)

    cer = 100.0 * total_char_ed / max(total_gt_chars, 1)
    ned = 100.0 * ned_sum / len(preds)
    wer = 100.0 * total_word_ed / max(total_ref_words, 1) if total_ref_words else float("nan")
    seq_acc = 100.0 * seq_correct / len(preds)
    return RecMetrics(cer=cer, ned=ned, wer=wer, seq_acc=seq_acc, n=len(preds))


def bootstrap_ci(
    preds: list[str],
    gts: list[str],
    metric: str = "cer",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a given metric."""
    rng = np.random.default_rng(seed)
    n = len(preds)
    if n == 0:
        return float("nan"), float("nan")
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p = [preds[i] for i in idx]
        g = [gts[i] for i in idx]
        m = compute_metrics(p, g)
        samples.append(getattr(m, metric))
    samples = np.array(samples)
    return float(np.percentile(samples, 100 * alpha / 2)), float(np.percentile(samples, 100 * (1 - alpha / 2)))
