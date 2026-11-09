"""Which error types does the language model remove? (seed 2, both test sets)

Classifies the edits of greedy and beam+LM decoding with the alignment of the error analysis
and vowel-order rule, and measures how uncertain the recognizer is per split
(share of non-blank frames whose runner-up class has probability above 0.05).
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
exec(open(HERE / "lm_decode.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0])
sys.path.insert(0, str(ROOT / "scripts"))
from diacritic_confusion import align  # noqa: E402  (same aligner as the error analysis)

LO, HI = 0x1200, 0x1357
def kind(a, b):
    ca, cb = ord(a), ord(b)
    if LO <= ca <= HI and LO <= cb <= HI:
        return "vowel" if (ca - LO) // 8 == (cb - LO) // 8 else "base"
    return "nonsyll"

voc = CTCVocab.from_json(); id2ch = {i + 1: ch for i, ch in enumerate(voc.chars)}
train = read_csv(D / "train/train_raw/image_text_pairs_train.csv")
vset = set(np.load(OUT / "val_idx.npy").tolist())
lm = CharKN([t for i, t in enumerate(train) if i not in vset], 6)
res = {}
for split, csvp in (("test_I", "test/test_rand/image_text_pairs_test_rand.csv"), ("test_II", "test/test_18th/image_text_pairs_test_18th.csv")):
    gt = read_csv(D / csvp)
    z = np.load(OUT / f"post_abl_asp_48x512_s2_{split}.npz"); V, I = z["vals"].astype(np.float32), z["ids"]
    nb = I[:, :, 0] != 0                      # frames whose top class is a character
    res[f"{split}|uncertain_frames"] = float((np.exp(V[:, :, 1]) > 0.05)[nb].mean() * 100)
    for name, dec in (("greedy", lambda i: greedy(V[i], I[i], id2ch)),
                      ("beam_lm", lambda i: beam_decode(V[i], I[i], id2ch, lm, 0.4, 1.0))):
        c = {"sub": 0, "ins": 0, "del": 0, "vowel": 0, "base": 0, "nonsyll": 0}
        for i, g in enumerate(gt):
            for op in align(g, dec(i)):
                t = op[0]
                if t in ("sub", "ins", "del"):
                    c[t] += 1
                    if t == "sub":
                        c[kind(op[1], op[2])] += 1
        res[f"{split}|{name}"] = c
        print(split, name, c, flush=True)
    print(split, "uncertain non-blank frames %.1f%%" % res[f"{split}|uncertain_frames"], flush=True)
json.dump(res, open(OUT / "lm_errors.json", "w"), indent=2)
