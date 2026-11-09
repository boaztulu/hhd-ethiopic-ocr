"""CPU inference jobs for the analyses.

1. Latency sweep (protocol of scripts/compute_cost.py: 4 threads, batch 32,
   5 warm-ups, median of 30) for DiffOCR at several refinement step counts,
   with CRNN-VGG timed in the same session as the reference.
2. CTC posteriors (top-16 log-probs per frame) of the three aspect-preserved
   CRNN-VGG seeds on the validation split, Test-I and Test-II, for
   language-model decoding. Greedy CER on CPU is logged as a consistency check.
3. DiffOCR accuracy at several refinement step counts on Test-I and Test-II,
   reusing one encoder pass per image across all step counts.
Images are read from local copies of the distributed zips.
"""
import csv, io, json, math, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image

from src.data.transforms import base_transform
from src.data.vocab import CTCVocab, Vocab
from src.eval import build_model
from src.utils.metrics import compute_metrics
sys.path.insert(0, str(ROOT / "scripts"))
from compute_cost import time_model  # noqa: E402

HERE = Path(__file__).parent
OUT = ROOT / "results" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
ZIPS = ROOT / "HHD-Ethiopic-data"
D = ROOT / "HHD-Ethiopic-data"
STEPS = [1, 2, 4, 8, 12, 16]
LOG = open(OUT / "cpu_jobs.log", "a", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def read_csv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 2:
                rows.append((r[0].strip(), r[1].rstrip(" ").rstrip()))
    return rows


def split_rows():
    train = read_csv(D / "train/train_raw/image_text_pairs_train.csv")
    rng = np.random.default_rng(0)  # val_seed in every config
    idx = np.arange(len(train)); rng.shuffle(idx)
    val_idx = idx[: int(round(0.10 * len(train)))]
    return {
        "val": ([train[i] for i in val_idx], ZIPS / "train_raw.zip", "train_raw/image_train/"),
        "test_I": (read_csv(D / "test/test_rand/image_text_pairs_test_rand.csv"), ZIPS / "test.zip", "test/test_rand/image_rand/"),
        "test_II": (read_csv(D / "test/test_18th/image_text_pairs_test_18th.csv"), ZIPS / "test.zip", "test/test_18th/image_18th/"),
    }, val_idx


def batches(rows, zpath, prefix, tf, bs):
    with zipfile.ZipFile(zpath) as zf:
        for s in range(0, len(rows), bs):
            chunk = rows[s:s + bs]
            imgs = [tf(image=np.array(Image.open(io.BytesIO(zf.read(prefix + fn))).convert("RGB")))["image"] for fn, _ in chunk]
            yield torch.stack(imgs), [t for _, t in chunk]


def load(run):
    ck = torch.load(ROOT / f"checkpoints/{run}/best.pt", map_location="cpu", weights_only=False)
    m, kind, cfg = build_model(ck, CTCVocab.from_json().n_classes, len(Vocab.from_json()))
    m.load_state_dict(ck["model"], strict=True)
    return m.eval(), kind, cfg


@torch.no_grad()
def refine(model, memory, T):
    """DiffOCR.generate with a precomputed encoder output and T refinement steps."""
    cfg = model.cfg
    B, L = memory.size(0), cfg.max_len
    x = torch.full((B, L), cfg.mask_id, dtype=torch.long)
    for step_idx in range(T):
        t_emb = torch.full((B,), int((1 - step_idx / max(1, T)) * (cfg.n_diffusion_steps - 1)), dtype=torch.long)
        probs = model.decoder(x, memory, t_emb).softmax(-1)
        top_probs, top_ids = probs.max(-1)
        if step_idx == T - 1:
            x = torch.where(x == cfg.mask_id, top_ids, x)
            break
        n_keep = max(0, int(L * math.cos((step_idx + 1) / T * math.pi / 2)))
        cur = x == cfg.mask_id
        score = torch.where(cur, -top_probs, torch.full_like(top_probs, -float("inf")))
        still = torch.zeros_like(cur)
        if n_keep > 0:
            still.scatter_(-1, torch.topk(score, k=n_keep, dim=-1).indices, True)
        x = torch.where(cur & ~still, top_ids, x)
    if (x == cfg.mask_id).any():
        logits = model.decoder(x, memory, torch.zeros(B, dtype=torch.long))
        x = torch.where(x == cfg.mask_id, logits.argmax(-1), x)
    return x


def latency():
    res = {}
    for label, run, name in (("CRNN-VGG", "abl_asp_48x512_s2", "crnn"), ("CRNN-ConvNeXt", "crnn_convnext_s1", "crnn"),
                             ("SVTR-Small", "svtr_small_s1", "svtr")):
        m, _, _ = load(run)
        res[label] = time_model(m, name, 48, 512, 32, 4)
        log("latency", label, res[label])
    diff, _, _ = load("diff_ocr_s1")
    for k in STEPS:
        diff.cfg.inference_steps = k
        res[f"DiffOCR_{k}"] = time_model(diff, "diff_ocr", 48, 512, 32, 4)
        log("latency", k, res[f"DiffOCR_{k}"])
    diff.cfg.inference_steps = 12
    json.dump(res, open(OUT / "latency.json", "w"), indent=2)


@torch.no_grad()
def dump_ctc(splits, runs):
    voc = CTCVocab.from_json()
    torch.set_num_threads(20)
    summary = {}
    for run in runs:
        model, _, _ = load(run)
        tf = base_transform(48, 512, preserve_aspect=True)
        for split, (rows, zpath, prefix) in splits.items():
            dest = OUT / f"post_{run}_{split}.npz"
            if dest.exists():
                log("skip", dest.name); continue
            t0 = time.time()
            vals, ids, preds, gts = [], [], [], []
            for x, texts in batches(rows, zpath, prefix, tf, 96):
                logp = model(x).permute(1, 0, 2)  # (B, T, C)
                v, i = logp.topk(16, dim=-1)
                vals.append(v.half().numpy()); ids.append(i.short().numpy())
                for row in logp.argmax(-1).numpy():
                    preds.append(voc.decode_greedy(row.tolist()))
                gts.extend(texts)
            np.savez_compressed(dest, vals=np.concatenate(vals), ids=np.concatenate(ids))
            m = compute_metrics(preds, gts)
            summary[f"{run}|{split}"] = {"cer": m.cer, "ned": m.ned, "secs": time.time() - t0}
            log(run, split, f"greedy CPU CER={m.cer:.3f} NED={m.ned:.3f}  ({time.time()-t0:.0f}s)")
            json.dump(summary, open(OUT / f"ctc_greedy_{run}.json", "w"), indent=2)


def diff_sweep(splits):
    voc = Vocab.from_json()
    torch.set_num_threads(20)
    model, _, cfg = load("diff_ocr_s1")
    tf = base_transform(48, 512, preserve_aspect=True)
    res = {}
    for split in ("test_I", "test_II"):
        rows, zpath, prefix = splits[split]
        preds = {k: [] for k in STEPS}; gts = []
        t0 = time.time()
        with torch.no_grad():
            for x, texts in batches(rows, zpath, prefix, tf, 64):
                memory = model.encoder(x)
                for k in STEPS:
                    preds[k].extend(voc.decode(s) for s in refine(model, memory, k).tolist())
                gts.extend(texts)
        for k in STEPS:
            m = compute_metrics(preds[k], gts)
            res[f"{split}|{k}"] = {"cer": m.cer, "ned": m.ned, "wer": m.wer}
            log("diffocr", split, k, f"CER={m.cer:.3f} NED={m.ned:.3f}")
            with open(OUT / f"diff_steps_{k}_{split}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["ground_truth", "prediction"]); w.writerows(zip(gts, preds[k]))
        log("diffocr", split, f"done in {time.time()-t0:.0f}s")
        json.dump(res, open(OUT / "diff_steps.json", "w"), indent=2)


if __name__ == "__main__":
    what = sys.argv[1:] or ["latency", "ctc", "diff"]
    splits, val_idx = split_rows()
    np.save(OUT / "val_idx.npy", val_idx)
    log("val lines", len(splits["val"][0]), "first idx", val_idx[:5].tolist())
    # heavy jobs first on all threads; latency last, alone, on 4 threads
    if "ctc" in what:
        dump_ctc(splits, ["abl_asp_48x512_s2", "abl_asp_48x512_s1", "abl_asp_48x512_s3"])
    if "diff" in what:
        diff_sweep(splits)
    if "latency" in what:
        latency()
    log("ALL DONE")
