"""Final language-model evaluation for the three aspect-preserving CRNN-VGG seeds.

Decodes Test-I and Test-II with greedy search, plain prefix beam search and
beam search with the character 6-gram, using the (alpha, beta) selected on the
validation split. Reports CER/NED/WER/line accuracy, word-space and full-stop
deletions, and paired bootstrap intervals for the gain (page-sized blocks on
Test-II, lines on Test-I).

usage: python lm_eval.py <alpha> <beta>
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
exec(open(HERE / "lm_decode.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0])
from src.utils.metrics import compute_metrics  # noqa: E402

SEEDS = ["abl_asp_48x512_s1", "abl_asp_48x512_s2", "abl_asp_48x512_s3"]
PUNCT = {"፡": "wordspace", "።": "fullstop"}


def align_dels(g, p):
    """Deletions of punctuation marks along one Wagner-Fischer alignment."""
    m, n = len(g), len(p)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32); dp[:, 0] = np.arange(m + 1); dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + (g[i - 1] != p[j - 1]))
    out, i, j = {k: 0 for k in PUNCT.values()}, m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (g[i - 1] != p[j - 1]):
            i -= 1; j -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            j -= 1
        else:
            if g[i - 1] in PUNCT:
                out[PUNCT[g[i - 1]]] += 1
            i -= 1
    return out


def blocks(n, b, rng):
    k = int(np.ceil(n / b)); s = rng.integers(0, n, size=k)
    return ((s[:, None] + np.arange(b)[None, :]) % n).ravel()[:n]


if __name__ == "__main__":
    alpha, beta = float(sys.argv[1]), float(sys.argv[2])
    voc = CTCVocab.from_json(); id2ch = {i + 1: ch for i, ch in enumerate(voc.chars)}
    train = read_csv(D / "train/train_raw/image_text_pairs_train.csv")
    vset = set(np.load(OUT / "val_idx.npy").tolist())
    lm = CharKN([t for i, t in enumerate(train) if i not in vset], 6)
    gts = {"test_I": read_csv(D / "test/test_rand/image_text_pairs_test_rand.csv"),
           "test_II": read_csv(D / "test/test_18th/image_text_pairs_test_18th.csv")}
    res, eds = {}, {}
    for run in SEEDS:
        for split, gt in gts.items():
            z = np.load(OUT / f"post_{run}_{split}.npz"); V, I = z["vals"].astype(np.float32), z["ids"]
            dec = {"greedy": [greedy(V[i], I[i], id2ch) for i in range(len(V))],
                   "beam": [beam_decode(V[i], I[i], id2ch, None, 0.0, 0.0) for i in range(len(V))],
                   "beam_lm": [beam_decode(V[i], I[i], id2ch, lm, alpha, beta) for i in range(len(V))]}
            for k, preds in dec.items():
                m = compute_metrics(preds, gt)
                r = {"cer": m.cer, "ned": m.ned, "wer": m.wer, "seq_acc": m.seq_acc}
                if split == "test_II" and k in ("greedy", "beam_lm"):
                    tot = {v: 0 for v in PUNCT.values()}
                    for g_, p_ in zip(gt, preds):
                        for a, b in align_dels(g_, p_).items():
                            tot[a] += b
                    r.update({f"del_{a}": b for a, b in tot.items()})
                res[f"{run}|{split}|{k}"] = r
                eds[(run, split, k)] = np.array([editdistance.eval(p, g) for p, g in zip(preds, gt)], float)
                print(run, split, k, json.dumps({a: round(b, 3) for a, b in r.items()}), flush=True)
    # paired bootstrap of the LM gain on the three-seed mean
    rng = np.random.default_rng(11)
    for split, b in (("test_I", 1), ("test_II", 50)):
        L = np.array([len(g) for g in gts[split]], float); n = len(L)
        for k in ("beam", "beam_lm"):
            d = []
            for _ in range(2000):
                idx = blocks(n, b, rng)
                f = lambda kk: np.mean([100 * eds[(r, split, kk)][idx].sum() / L[idx].sum() for r in SEEDS])
                d.append(f(k) - f("greedy"))
            res[f"gain_{k}_vs_greedy|{split}"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            print(split, k, "vs greedy 95% interval", res[f"gain_{k}_vs_greedy|{split}"], flush=True)
    res["alpha"], res["beta"] = alpha, beta
    json.dump(res, open(OUT / "lm_eval.json", "w"), indent=2)
