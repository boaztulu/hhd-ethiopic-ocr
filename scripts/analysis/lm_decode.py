"""Character n-gram language model + CTC prefix beam search (shallow fusion).

The LM is an interpolated Kneser-Ney character model trained on the training
transcriptions minus the validation lines. Beam width, LM weight (alpha) and
length bonus (beta) are chosen on the validation split; Test-I and Test-II are
decoded once with the selected setting. Greedy decoding of the same posteriors
is the paired baseline.
"""
import csv, json, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import editdistance

ROOT = Path(__file__).resolve().parents[2]  # project root
sys.path.insert(0, str(ROOT))
from src.data.vocab import CTCVocab  # noqa: E402

HERE = Path(__file__).parent
OUT = ROOT / "results" / "analysis"
D = ROOT / "HHD-Ethiopic-data"
BOS, EOS = "<s>", "</s>"


def read_csv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 2:
                rows.append(r[1].rstrip(" ").rstrip())
    return rows


class CharKN:
    """Interpolated Kneser-Ney over characters, uniform floor below the unigram."""

    def __init__(self, texts, order):
        self.N = order
        raw = defaultdict(lambda: defaultdict(int))          # highest order: raw counts
        for t in texts:
            s = [BOS] * (order - 1) + list(t) + [EOS]
            for i in range(order - 1, len(s)):
                raw[tuple(s[i - order + 1:i])][s[i]] += 1
        self.vocab = sorted({c for t in texts for c in t}) + [EOS]
        self.V = len(self.vocab) + 1                           # +1 reserves mass for unseen chars
        # counts[k][ctx][w]: k = context length; raw at k = N-1, continuation counts below
        self.counts = {order - 1: raw}
        for k in range(order - 2, -1, -1):
            cont = defaultdict(lambda: defaultdict(int))
            for ctx, nxt in self.counts[k + 1].items():
                for w in nxt:
                    cont[ctx[1:]][w] += 1
            self.counts[k] = cont
        self.tot, self.n1p, self.D = {}, {}, {}
        for k, table in self.counts.items():
            n1 = n2 = 0
            for ctx, nxt in table.items():
                self.tot[(k, ctx)] = sum(nxt.values())
                self.n1p[(k, ctx)] = len(nxt)
                for c in nxt.values():
                    n1 += c == 1; n2 += c == 2
            self.D[k] = n1 / (n1 + 2 * n2) if n1 + 2 * n2 else 0.5
        self.cache = {}

    def prob(self, ctx, w):
        """P(w | ctx) with ctx the last N-1 symbols (BOS-padded)."""
        key = (ctx, w)
        p = self.cache.get(key)
        if p is not None:
            return p
        p = 1.0 / self.V
        for k in range(0, self.N):
            h = ctx[len(ctx) - k:] if k else ()
            tot = self.tot.get((k, h))
            if not tot:
                break
            c = self.counts[k][h].get(w, 0)
            d = self.D[k]
            p = (max(c - d, 0) + d * self.n1p[(k, h)] * p) / tot
        if len(self.cache) > 3_000_000:
            self.cache.clear()
        self.cache[key] = p
        return p

    def logprob(self, ctx, w):
        return math.log(self.prob(ctx, w))

    def ppl(self, texts):
        lp, n = 0.0, 0
        for t in texts:
            s = [BOS] * (self.N - 1) + list(t) + [EOS]
            for i in range(self.N - 1, len(s)):
                lp += self.logprob(tuple(s[i - self.N + 1:i]), s[i]); n += 1
        return math.exp(-lp / n)


NEG = -1e30


def lse(a, b):
    if a < b:
        a, b = b, a
    if b <= NEG:
        return a
    return a + math.log1p(math.exp(b - a))


def beam_decode(vals, ids, id2ch, lm, alpha, beta, beam=10, cutoff=math.log(1e-3)):
    """CTC prefix beam search; vals/ids are (T, K) top-K log-probs and class ids (0 = blank)."""
    N1 = lm.N - 1 if lm else 0
    beams = {(): (0.0, NEG)}                          # prefix -> (log p_blank, log p_nonblank)
    lmscore = {(): 0.0}                               # prefix -> accumulated alpha * log P_lm
    for t in range(vals.shape[0]):
        v, ix = vals[t], ids[t]
        blank = NEG
        cands = []
        for k in range(v.shape[0]):
            lpk = float(v[k])
            if ix[k] == 0:
                blank = lpk
            elif lpk > cutoff:
                cands.append((int(ix[k]), lpk))
        nb = defaultdict(lambda: [NEG, NEG])
        for pre, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            e = nb[pre]; e[0] = lse(e[0], tot + blank)
            last = pre[-1] if pre else None
            for c, lpc in cands:
                if c == last:
                    e[1] = lse(e[1], pnb + lpc)
                    src = pb + lpc
                else:
                    src = tot + lpc
                new = pre + (c,)
                if new not in lmscore:
                    if lm:
                        ctx = tuple(([BOS] * N1 + [id2ch[i] for i in pre])[-N1:]) if N1 else ()
                        lmscore[new] = lmscore[pre] + alpha * lm.logprob(ctx, id2ch[c])
                    else:
                        lmscore[new] = 0.0
                f = nb[new]; f[1] = lse(f[1], src)
        scored = sorted(nb.items(), key=lambda kv: lse(*kv[1]) + lmscore[kv[0]] + beta * len(kv[0]), reverse=True)
        beams = {p: tuple(s) for p, s in scored[:beam]}

    def final(p):
        s = lse(*beams[p]) + lmscore[p] + beta * len(p)
        if lm:
            ctx = tuple(([BOS] * N1 + [id2ch[i] for i in p])[-N1:]) if N1 else ()
            s += alpha * lm.logprob(ctx, EOS)
        return s
    best = max(beams, key=final)
    return "".join(id2ch[i] for i in best)


def greedy(vals, ids, id2ch):
    top = ids[:, 0]
    out, prev = [], None
    for c in top:
        c = int(c)
        if c != prev and c != 0:
            out.append(id2ch[c])
        prev = c
    return "".join(out)


def cer(preds, gts):
    return 100.0 * sum(editdistance.eval(p, g) for p, g in zip(preds, gts)) / sum(len(g) for g in gts)


if __name__ == "__main__":
    voc = CTCVocab.from_json()
    id2ch = {i + 1: ch for i, ch in enumerate(voc.chars)}
    train = read_csv(D / "train/train_raw/image_text_pairs_train.csv")
    val_idx = np.load(OUT / "val_idx.npy")
    vset = set(val_idx.tolist())
    lm_text = [t for i, t in enumerate(train) if i not in vset]
    val_gt = [train[i] for i in val_idx]
    gt = {"val": val_gt,
          "test_I": read_csv(D / "test/test_rand/image_text_pairs_test_rand.csv"),
          "test_II": read_csv(D / "test/test_18th/image_text_pairs_test_18th.csv")}
    mode = sys.argv[1]
    if mode == "ppl":
        for n in (2, 3, 4, 5, 6, 7, 8):
            lm = CharKN(lm_text, n)
            print(f"order {n}: val ppl {lm.ppl(val_gt):.2f}  test_I {lm.ppl(gt['test_I']):.2f}  test_II {lm.ppl(gt['test_II']):.2f}", flush=True)
    elif mode == "tune":
        run, order, nval = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        lm = CharKN(lm_text, order)
        z = np.load(OUT / f"post_{run}_val.npz"); V, I = z["vals"].astype(np.float32), z["ids"]
        sel = np.random.default_rng(0).permutation(len(val_gt))[:nval]
        g = [val_gt[i] for i in sel]
        base = cer([greedy(V[i], I[i], id2ch) for i in sel], g)
        res = {"greedy": base}
        print(f"{run} val subset n={nval} greedy {base:.3f}", flush=True)
        for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            for beta in ((0.0,) if alpha == 0 else (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)):
                t0 = time.time()
                p = [beam_decode(V[i], I[i], id2ch, lm if alpha > 0 else None, alpha, beta) for i in sel]
                res[f"{alpha}|{beta}"] = cer(p, g)
                print(f"  alpha={alpha} beta={beta}: {res[f'{alpha}|{beta}']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT / f"lm_tune_{run}_o{order}.json", "w"), indent=2)
    elif mode == "test":
        run, order, alpha, beta = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        lm = CharKN(lm_text, order)
        res = {}
        for split in ("val", "test_I", "test_II"):
            z = np.load(OUT / f"post_{run}_{split}.npz"); V, I = z["vals"].astype(np.float32), z["ids"]
            t0 = time.time()
            pg = [greedy(V[i], I[i], id2ch) for i in range(len(V))]
            pb = [beam_decode(V[i], I[i], id2ch, None, 0.0, 0.0) for i in range(len(V))]
            pl = [beam_decode(V[i], I[i], id2ch, lm, alpha, beta) for i in range(len(V))]
            res[split] = {"greedy": cer(pg, gt[split]), "beam": cer(pb, gt[split]), "beam_lm": cer(pl, gt[split]),
                          "secs": time.time() - t0}
            with open(OUT / f"lm_preds_{run}_{split}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["ground_truth", "greedy", "beam", "beam_lm"])
                w.writerows(zip(gt[split], pg, pb, pl))
            print(run, split, json.dumps(res[split]), flush=True)
        json.dump(res, open(OUT / f"lm_test_{run}.json", "w"), indent=2)
