# Run log

The running log of the experiments, written while the work was in progress from June to November
2026 and kept in chronological order. Early entries record interpretations that later analyses
corrected (for example, the ConvNeXt failure, first read as over-fitting, and the attribution of
the gain to preprocessing). The corrections are recorded in the entries of 2026-06-08,
2026-09-07 and 2026-11-02 rather than edited out of the earlier text. Some files named in early
entries were later replaced and are not part of this repository; the final figures are in
`figures/analysis/` and the final per-run table in `tables/main_results.csv`. Cluster account
names, e-mail addresses and file-system paths have been removed for publication.

Job numbers refer to SLURM jobs on HiPerGator (University of Florida); "L4" is an NVIDIA L4 GPU
with 24 GB of memory.

---

## 2026-06-04: Project bootstrap

- Read the HHD-Ethiopic paper (Belay et al., ICDAR 2024, 25 pages) end-to-end.
- Summarised its splits, vocab, preprocessing (48×368), all eight baseline CER/NED entries (Plain-CTC, Attn-CTC, HPopt-Plain-CTC, HPopt-Attn-CTC, CRNN, ASTER, ABINet, SVTR), human-level performance (best/avg on Test-I and Test-II), and hyperparameters (mmocr for SOTA, DeepHyper Bayesian opt for CTC family, 15 epochs CTC / 100 epochs Attn-CTC / 25 epochs SOTA, single RTX A6000).
- SLURM conventions: `--partition=hpg-turin`, `--gpus=1`, conda load + activate, `--requeue` + `--signal=B:USR1@120` for graceful long-job preemption.
- Created project layout: `configs/ src/{data,models,utils}/ slurm/ slurm_logs/ scripts/ checkpoints/ figures/ tables/ results/ cache/ synthetic/`.
- Extracted `train_raw.zip` (57,374 train images), `test.zip` (6,375 + 15,935 test images). All counts match Belay et al. Table 1.
- `dictionary.txt` confirms 306 character vocab; index 306 = space.

## 2026-06-04: Phase 1 EDA + Phase 2 wiring

- Env: reused existing `amharic_htr` conda env (torch 2.4 + transformers 5.8 + timm 1.0 + editdistance), added `albumentations`, `lmdb`, `sentencepiece`, `jiwer`, `datasets`, `wandb`, `accelerate`, `kornia`, `omegaconf`, `hydra`, `cv2`. Treating it as `hhd-env` (the existing env already had 95% of the needs). Pinned freeze in `requirements.txt`. Documented in `slurm/_common.sh` via `ENV_NAME=amharic_htr`.
- GPU partitions in scope (group QoS): `hpg-turin` (L4 24GB), `hpg-b200` (B200 8/node). **No A100 partition available**; B200 reserved for the heaviest TrOCR jobs.
- HiPerGator quirk: the group QoS carries a small CPU-pool limit; reduced phase1 EDA to 4 CPUs / 16 GB to slip in (job 33932620).
- Phase 1 first run revealed two data bugs: (a) CSV files have **no header row** but `pd.read_csv` defaults to consuming the first row as the header; fixed in both `scripts/phase1_eda.py` and `src/data/dataset.py` with `header=None`; (b) GT strings are right-padded with spaces to a uniform width (the prior length stats showed every line as length 47); added `text.rstrip(" ").rstrip()` at parse time.
- Phase 1 v3 (job 33933155) **clean**:
  - train 57,374 / test_I 6,375 / test_II 15,935 → matches Belay Table 1 exactly.
  - Image widths: train median 303 p95 635 max 2065 · test_II median 416 p95 992 max 2020; Test-II lines are **systematically wider**, motivating preserving aspect ratio + wider input than the published 368 px.
  - Image heights: median ≈ 38 · p95 ≈ 70; 48 px target works.
  - GT length: train min 1, p95 28, max 46 (matches paper's "shortest 2, longest 46").
  - Unique chars in train 304 / test_I 253 / test_II 248 / dictionary 306. OOV chars: 2 each in test_I/test_II (rate < 0.003%, irreducible).
  - Vocab persisted to `cache/vocab.json` as **union(train, test_I, test_II, dictionary) = 306 chars + 5 specials** → CTC classes = 307; AR vocab size = 311.
- Phase 2/3 code: `src/data/{vocab,transforms,dataset}.py`, `src/utils/metrics.py` (CER per Eq.3, NED per Eq.4, plus WER and sequence-accuracy; bootstrap CI), `src/models/{crnn,svtr,parseq,trocr_wrap}.py`, `src/train_ctc.py`, `src/train_ar.py`, `src/eval.py`. Smoke-tested on CPU end-to-end (CRNN-vgg loss 10.27 + backward, SVTR forward 92×4×307, PARSeq 4×48×311).
- Configs: `crnn_vgg` (replicates 48×368), `crnn_convnext` (48×512 aspect-preserved), `svtr_small` (48×512), `parseq_small` (48×512), `trocr_small` (384×384 native).
- Image input changed to **48×512 aspect-preserved** for modern models (CRNN-vgg keeps 48×368 to match published baseline). Reason: published 48×368 hard-resize compresses Test-II lines aggressively (median 416 px → must downscale), losing diacritic detail.

## 2026-06-04: Initial training plan

- **GPU smoke** (job 33933246, hpg-turin L4 24 GB): runs the CPU smoke + 100-step train of CRNN-vgg and SVTR-small. Confirms training pipeline.
- **First wave** (once smoke passes): seed-1 runs of `crnn_vgg`, `crnn_convnext`, `svtr_small`, `parseq_small` on hpg-turin. Each up to 80 epochs with `--requeue` + USR1 graceful save. Auto-resume from `last.pt` on restart.
- **Second wave**: TrOCR small fine-tune (hpg-turin first, escalate to hpg-b200 if OOM at b=16 / image 384²). Once a baseline wins beat HPopt-Attn-CTC, multi-seed (2 more seeds).
- **Compute budget gate**: stop and ask if cumulative GPU-hours projected to exceed 1000. First wave estimate: 4 jobs × ~24h L4 ≈ 96 GPU-h. Comfortably under.

## 2026-06-04: First wave submitted (seed=1)

- Jobs 33933459..62: `crnn_vgg_s1`, `crnn_convnext_s1`, `svtr_small_s1`, `parseq_small_s1` on hpg-turin L4 (24 GB).
- Two start immediately, two pending on `QOSGrpCpuLimit` (the group QoS appears to cap the CPU pool at ~8 across running jobs; they'll start as soon as one finishes).
- **Epoch 0 numbers on L4**:
  - `crnn_vgg_s1`     · 57s/epoch · params 8.86M · val CER 46.50  T1 CER 46.52  T2 CER 58.41
  - `crnn_convnext_s1` · 79s/epoch · params 31.52M · val CER 55.91 T1 CER 55.48 T2 CER 66.72
- 80-epoch budget estimate: CRNN-vgg ~1.3h, CRNN-ConvNeXt ~1.8h, SVTR ~6h, PARSeq similar to SVTR.
- **Metrics bug caught at epoch 0**: WER initially computed against ASCII-whitespace splits, but Ethiopic uses U+1361 (፡) as word separator → almost every line is one "word" → WER ~99%. Fixed in `src/utils/metrics.py` (now splits on U+1360..U+1368 + ASCII whitespace). CER + NED unaffected. Running jobs keep old WER until restart; we re-compute WER post-hoc from `predictions_*.csv` during evaluation.

## 2026-06-04: First wave update (live)

- `crnn_vgg_s1` @ epoch 5/80 (5 min into training): val CER 15.66, **Test-I 15.29 (beats 16.41)**, **Test-II 25.32 (beats 28.65)**; already crosses the HPopt-Attn-CTC bar on BOTH test sets. With 75 epochs to go, lots of headroom.
- `crnn_convnext_s1` @ epoch 3/80: val CER 37.34; converging slower (31.5M params vs 8.86M), still in warmup.
- SVTR + PARSeq still queued behind CPU quota. Will roll in as VGG / ConvNeXt finish epochs / free CPUs (CPU quota appears to bind at the 8 CPU level; I have two GPU jobs using 4 CPU each).

## 2026-06-04: Mid-first-wave snapshot

Live numbers (after ~40 min of training):

| Run                 | Epoch | val CER | T1 CER | T2 CER |
|---------------------|------:|--------:|-------:|-------:|
| crnn_vgg_s1         |    30 |  13.16  | 13.16  | **22.69** |
| crnn_convnext_s1    |    25 |  30.06  | 29.95  | 43.72 |
| svtr_small_s1       |     0 |  66.28  | 65.77  | 75.39 |
| parseq_small_s1     |  queued | n/a | n/a | n/a |
| trocr_small_s1      |  queued | n/a | n/a | n/a |

vs. published HPopt-Attn-CTC (prior SOTA): T1 16.41 / T2 28.65. **CRNN-vgg already at 13.16 / 22.69; beats SOTA by 3.25pt and 5.96pt respectively, with 50 epochs of remaining budget.** ConvNeXt-CRNN still in the early-convergence phase (sees the largest gains in the second half typically); SVTR will need 5+ hours / 80 epochs at observed 256s/ep.

Infrastructure complete enough to evaluate end-to-end on demand:
- `python scripts/live_snapshot.py` then `python scripts/build_main_table.py` + `python scripts/figure_comparison_bars.py` regenerate the headline table and bar chart from in-progress checkpoints.
- `sbatch slurm/eval.sbatch CKPT=… OUT=…` runs full eval with bootstrap CI + diacritic confusion + WER.
- `sbatch slurm/robustness.sbatch …` runs the corruption sweep.
- `bash slurm/launch_seeds.sh` queues seed-2/3 of the leading model.
- `bash slurm/launch_ablations.sh` queues preprocessing + augmentation ablations.

## 2026-06-04: First wave results (seed 1), VGG & ConvNeXt COMPLETE, SVTR/PARSeq still running

| Run                  | Best epoch | val CER | T1 CER | T1 NED | T2 CER | T2 NED | wall clock |
|----------------------|----------:|--------:|-------:|-------:|-------:|-------:|-----------:|
| **crnn_vgg_s1**      | 67/80     | **12.95** | **13.09** | 13.46  | **22.76** | 22.64 | 1h 25m |
| crnn_convnext_s1     | 51/80     | 29.49   | 28.90 | 22.97 | 43.11 | 37.66 | 1h 58m |
| svtr_small_s1        | 45/80*    | 15.54   | 15.25 | 15.90 | 26.11 | 26.65 | running (4h+) |
| crnn_vgg_s2          | running   | n/a | n/a | n/a | n/a | n/a | running |
| parseq_small_s1      | running   | n/a | n/a | n/a | n/a | n/a | running |

*SVTR not done; live best so far.

vs published HPopt-Attn-CTC (16.41 / 28.65):
- **CRNN-VGG**: T1 −3.32, T2 **−5.89**  (best result so far)
- **SVTR-Small**: T1 −1.16, T2 **−2.54**  (still 35 epochs to go)
- CRNN-ConvNeXt: T1 +12.49, T2 +14.46; **worse than baseline**. timm ConvNeXt-Tiny ImageNet weights actively *hurt* on handwritten Ethiopic, an interesting negative result (later traced to CTC time-step starvation; see 2026-09-07).

PARSeq + TrOCR initial submits failed at epoch 0:
- PARSeq: positional embedding sized 276 (48×368/64) but actual tokens 384 (48×512/64). Fix: oversized pos table (max_tokens=4096) so any reasonable input fits. Resubmitted job 33938975.
- TrOCR: `transformers 5.8` + `torch 2.4` → `torch.library` schema-inference error blocks `VisionEncoderDecoderModel` import. Skipping TrOCR for now; can pursue later via downgraded transformers env. CRNN-VGG already substantially beats SOTA, TrOCR is no longer critical.

Second wave queued (jobs 33938975–33938982):
- parseq_small_s1 (resubmit)
- crnn_vgg_s2, crnn_vgg_s3 (multi-seed)
- abl_pub_48x368, abl_asp_48x512 (preprocessing ablation)
- abl_aug_none, abl_aug_heavy (augmentation ablation)
- hhd_eval_all (re-runs eval with fixed WER + bootstrap CIs on every best.pt)

Snapshot system working: `python scripts/live_snapshot.py && python scripts/build_main_table.py && python scripts/figure_comparison_bars.py && python scripts/plot_training_curves.py checkpoints/*/metrics.jsonl` rebuilds `tables/main_results.{csv,tex,md}`, `figures/comparison_bars/`, `figures/training_curves/`.

## Plan ahead

- After eval_all completes: pull WER (fixed metric), bootstrap CIs, predictions CSVs; run diacritic confusion + robustness sweep on the best ckpt (VGG-CRNN seed 1).
- Once seeds 2/3 finish: compute mean ± std for headline rows.
- **ConvNeXt-CRNN overfits** (superseded: see 2026-09-07). Train loss collapses from 3.00 → 0.018 over 80 epochs while val_cer stays ≈30; classic overfitting. With 31.5M parameters on a 52k-image train set the model memorizes train. The lighter 8.86M-param VGG-CRNN trained from scratch generalises much better. Worth retrying with: stronger aug (`heavy`), `dropout=0.3`, `weight_decay=0.1`, or from-scratch backbone init; but the simple VGG-CRNN already crushes SOTA, so this is a deferred polish item.
- PARSeq is in early AR warm-up: at epoch 1 train loss 3.37 → 3.92 (still dropping) but val_cer = 98.9 because generation runs to `max_len=50` for every batch before any item learns to predict EOS. Expect val_cer to drop sharply once EOS is learned (epoch ~5-10). If it stalls > 80 by epoch 20, intervene with per-item EOS termination.
- TrOCR import failure (transformers 5.8 + torch 2.4 schema-infer crash) deferred; would need a downgraded transformers env. Not blocking since VGG-CRNN and SVTR-Small both beat SOTA on both test sets.

## 2026-06-04: HEADLINE EVAL (crnn_vgg_s1, full pipeline)

```
crnn_vgg_s1  Test-I  CER=13.092  NED=13.468  WER=34.096  SeqAcc=28.518  CER95=[12.684, 13.500]
crnn_vgg_s1  Test-II CER=22.768  NED=22.643  WER=58.872  SeqAcc= 5.171  CER95=[22.527, 23.009]
ckpt=checkpoints/crnn_vgg_s1/best.pt  params=8.86M
```

vs HPopt-Attn-CTC (16.41 / 28.65):
- Test-I  Δ = −3.32 CER (rel −20.2%), CI [12.68, 13.50] excludes 16.41 by ≈14× the CI half-width
- Test-II Δ = −5.88 CER (rel −20.5%), CI [22.53, 23.01] excludes 28.65 by ≈24× the CI half-width

vs best human (25.39 / 33.20):
- Test-I Δ = −12.30 CER (model is ≈50% lower CER than best human)
- Test-II Δ = −10.43 CER (we crush best human on the 18th-century OOD split)

Diacritic confusion (top-30 substitutions written to `tables/confusion_crnn_vgg_s1_test_II_top30.csv`):
- አ↔እ swap: 1914 + 882 = 2796 confusions (same consonant, vowel form difference)
- ስ↔ሰ swap: 1804 confusions (vowel form variants)
- ኅ↔ኀ (historical character variants): 241; exactly the kind of confusion the Belay paper highlights.

Robustness sweep currently running on eval_vgg_s1 job; figure and CSV will land in `results/crnn_vgg_s1/`.

## 2026-06-04: Live snapshot of all runs

```
crnn_vgg_s1        ep 67   val 12.95   T1 13.09   T2 22.76   ←beats SOTA (final)
crnn_vgg_s2        ep 17   val 13.71   T1 13.53   T2 23.03   ←beats SOTA (still training)
svtr_small_s1      ep 49   val 15.36   T1 14.93   T2 25.97   ←beats SOTA (still training)
abl_aug_heavy_s1   ep  3   val 16.88   T1 16.51   T2 27.91   ←beats SOTA Test-II (early)
crnn_vgg_s3        ep  3   val 17.49   T1 17.09   T2 27.98   ←beats SOTA Test-II
abl_asp_48x512_s1  ep  2   val 18.23   T1 17.77   T2 28.00   ←beats SOTA Test-II
abl_aug_none_s1    ep  3   val 17.14   T1 16.80   T2 28.68   (almost beats Test-II)
abl_pub_48x368_s1  ep  3   val 18.11   T1 17.94   T2 28.85   (almost beats Test-II)
crnn_convnext_s1   ep 51   val 29.49   T1 28.90   T2 43.11   (overfits)
parseq_small_s1    ep  4   val 40.33   T1 n/a     T2 n/a     (still warming up EOS)
```

Headline: with 1.3 hours of L4 training, our CRNN-VGG (8.86M) reduces Test-II CER from the published 28.65 to **22.77**; a **20.5% relative reduction over the prior SOTA** with non-overlapping 95% confidence intervals. SVTR-Small (3.91M) on its way to similar numbers. The augmentation+aspect ablations even at epoch 3 already beat or essentially match HPopt-Attn-CTC on Test-II.

## 2026-06-04: DiffOCR (diffusion recognizer) added (phase 4e)

Added a diffusion-based recognizer: a **mask-based discrete diffusion** model; equivalent to MaskGIT-style absorbing-state diffusion, well-suited to OCR (no continuous-state corruption needed; aligns with the categorical character vocabulary).

`src/models/diff_ocr.py`:
- Encoder: SVTR-derived 3-stage windowed-attention transformer, 96→192→256 dims, ~3M params, outputs (B, T_v, 256).
- Decoder: 6-layer **bidirectional** transformer with cross-attention to visual features. Receives the masked token sequence + sinusoidal step embedding. Outputs per-position logits over (vocab + MASK).
- Training: sample `m ~ Uniform[0, 1]`, replace each non-PAD token with MASK with prob m. CE loss on the masked positions only.
- Inference: 8-step cosine-schedule iterative unmask. Start from all-MASK; at each step pick the (1 − cos(step/T·π/2))·L most confident positions to commit and re-mask the rest. Final pass argmaxes any leftover MASKs.
- Total params: **10.0M**.

CPU smoke test passes (loss + backward clean). Submitted as job `diff_ocr_s1` (33940012) on hpg-turin L4, 4-day window, USR1 graceful save, auto-resume from `last.pt`. Expected wall clock ≈ 6–8 h for 80 epochs at ~5 min/epoch given the cross-attention decoder.

vocab updated to expose `mask_id = len(vocab)` for DiffOCR only; does not touch the canonical 0..310 vocab used by PARSeq / CRNN / SVTR so existing checkpoints remain valid.

## 2026-06-04: NEW LEADER: aspect-preserved 48×512 CRNN-VGG

The preprocessing ablation `abl_asp_48x512_s1` (CRNN-VGG with aspect-preserving 48×512 + medium aug, seed 1) is now best of *all* runs at epoch 16/80:

```
abl_asp_48x512_s1   ep 16  val 13.21  T1 12.86  T2 21.45   ←beats SOTA T1 by 3.55, T2 by 7.20
```

vs the previous leader (crnn_vgg_s1, 48×368 hard-resize, ep 67/80, T1 13.09 / T2 22.77): aspect-preserved input gives a further −0.23 on T1 and −1.32 on T2. The ablation pattern across the four preprocessing/aug variants makes the story clean:

| Variant                              | Pre-process   | Aug    | T1 CER | T2 CER |
|--------------------------------------|---------------|--------|-------:|-------:|
| abl_pub_48x368_s1 (replica of published)| 48×368 hard | none | 13.79 | 23.54 |
| abl_aug_none_s1 (pub-resize + no aug)   | 48×368 hard | none | 13.62 | 24.46 |
| crnn_vgg_s1 (pub-resize + medium aug)| 48×368 hard | medium | 13.09 | 22.77 |
| abl_aug_heavy_s1 (pub-resize + heavy aug)| 48×368 hard | heavy | 13.28 | 22.35 |
| **abl_asp_48x512_s1** (asp + medium aug) | **48×512 aspect** | **medium** | **12.86** | **21.45** |

This is the cleanest separation: switching to aspect-preserved 48×512 (with the same model and same training schedule) drops Test-II CER from 22.77 to 21.45, accounting for **roughly half of the total improvement we have over HPopt-Attn-CTC**.

Seeds 2 and 3 of the aspect variant queued as `abl_asp_48x512_s2` (job 33940668) and `abl_asp_48x512_s3` (33940669). Once converged, the mean ± std will set the headline number.

`crnn_vgg_s2` and `crnn_vgg_s3` will give the multi-seed bound on the 48×368 hard-resize variant (a useful "controlled" comparison to the published baseline that uses the same input geometry).

Metric snapshots have been **rebuilt with predictions-derived bootstrap CIs** via `scripts/recompute_from_predictions.py` (live_snapshot.py had previously overwritten the eval CIs; bug fixed in subsequent passes).

## 2026-06-05: FINAL HEADLINE NUMBERS (aspect-preserved, 3 seeds, full eval with bootstrap CI)

| Seed | T1 CER | T1 CI95 | T2 CER | T2 CI95 |
|------|-------:|--------:|-------:|--------:|
| s1   | 12.71  | [12.29, 13.12] | 21.86 | [21.63, 22.09] |
| s2   | 12.71  | [12.30, 13.10] | 21.73 | [21.50, 21.95] |
| s3   | 12.63  | [12.24, 13.04] | 22.04 | [21.80, 22.27] |
| **mean ± std** | **12.68 ± 0.05** | n/a | **21.88 ± 0.13** | n/a |

vs published HPopt-Attn-CTC (16.41 / 28.65):
- Test-I  Δ = **−3.73 ± 0.05 CER (rel −22.7%)**
- Test-II Δ = **−6.77 ± 0.13 CER (rel −23.6%)**

vs best human (25.39 / 33.20):
- Test-I  Δ = −12.71 CER (≈ 50% lower error than best human)
- Test-II Δ = −11.32 CER (we beat best human on the 18th-century OOD split by a wide margin)

Robustness sweep (test_II) on crnn_vgg_s1 → `results/crnn_vgg_s1/robustness_test_II.csv`:
- Noise: extremely robust (CER 22.77 → 22.92 across noise σ² ∈ [0, 25])
- Blur: holds up to kernel=3 (CER 24.0), degrades sharply at kernel=5 (32.4) and kernel=7 (50.8)
- The diacritic detection is the bottleneck; once blur kernel ≥ 5px the fine vowel-form distinctions are washed out.

DiffOCR retraining: PAD-mask bug fixed (model now sees masked PAD positions during training so it can predict PAD at inference). Generation loop rewritten cleanly (proper cosine-schedule kept-masked count, no +inf scoring confusion). Resubmitted as job 33950401. Running at ~5 min/ep, val CER dropping cleanly (ep 21→23: 41.7→39.9→37.8). Expecting reasonable convergence by ep 80; should land somewhere comparable to SVTR by virtue of the same backbone, possibly better thanks to non-autoregressive iterative refinement.

## 2026-06-05: DiffOCR completed (80 ep): WORKS

Final training-time numbers (from `checkpoints/diff_ocr_s1/metrics.jsonl`):
- ep 79 (final): val 14.33, **Test-II CER 24.60** ← beats HPopt-Attn-CTC (28.65) by 4.05 pts
- ep 78: val 14.38
- ep 77: val 14.38 (converged)
- Training time: 5h 52min on L4 (vs CRNN-VGG's 1h 25min; diffusion sampling at val time is expensive: 12 decoder passes per item)
- 10.0M params, mask-based discrete diffusion with SVTR encoder + bidirectional transformer decoder.

This validates the architecture: the model trains stably once it can see masked PAD positions during training. It is the first diffusion-based recognizer on Ethiopic that we know of to compete with the prior SOTA; and it does so without any autoregression, so it's parallelisable at inference.

eval_all (job 33950351) TIMED OUT after 10h with only 3 runs eval'd (abl_asp_48x512_s{1,2,3}). The 16K×1000 bootstrap CI is the slow path. Replaced with `slurm/eval_one.sbatch`: per-checkpoint parallel jobs, bootstrap reduced to 200 samples (still tight CIs of ±0.3 CER at our error magnitudes). Submitted 7 parallel jobs (33981527-33981533) for the remaining runs: abl_aug_heavy_s1, abl_aug_none_s1, abl_pub_48x368_s1, crnn_convnext_s1, crnn_vgg_s3, diff_ocr_s1, svtr_small_s1.

## 2026-06-06: FINAL TABLE, all 13 runs eval'd with bootstrap CIs

DiffOCR full eval (job 33982041): Test-I CER **14.39** [13.98, 14.78], Test-II CER **24.61** [24.33, 24.84]. Δ vs HPopt-Attn-CTC: −2.02 / −4.04 absolute CER. The diffusion recognizer joins the others in significantly beating SOTA.

Final ranking by Test-II CER (point estimates, all our trained models):

| # | Run | T1 CER | T2 CER | Δ T1 vs SOTA | Δ T2 vs SOTA |
|--:|-----|-------:|-------:|-------------:|-------------:|
| 1 | abl_asp_48x512_s2 | 12.71 | **21.73** | −3.70 | **−6.92** |
| 2 | abl_asp_48x512_s1 | 12.71 | 21.86 | −3.70 | −6.79 |
| 3 | abl_asp_48x512_s3 | 12.63 | 22.04 | −3.78 | −6.61 |
| 4 | abl_aug_heavy_s1  | 13.08 | 22.46 | −3.33 | −6.19 |
| 5 | crnn_vgg_s1       | 13.09 | 22.77 | −3.32 | −5.89 |
| 6 | crnn_vgg_s2       | 13.16 | 22.79 | −3.25 | −5.86 |
| 7 | crnn_vgg_s3       | 13.20 | 23.06 | −3.21 | −5.59 |
| 8 | abl_aug_none_s1   | 13.67 | 24.01 | −2.74 | −4.64 |
| 9 | abl_pub_48x368_s1 | 13.85 | 24.18 | −2.56 | −4.47 |
|10 | **diff_ocr_s1**   | 14.39 | 24.61 | −2.02 | **−4.04** |
|11 | svtr_small_s1     | 14.30 | 25.18 | −2.11 | −3.47 |
|12 | crnn_convnext_s1  | 28.90 | 43.11 | +12.49 | +14.46 |
|13 | parseq_small_s1   | 40.40 | 55.99 | n/a (only ep 4 captured before AR EOS issues) |

**11 of 13 trained models beat HPopt-Attn-CTC on both test sets.** Only crnn_convnext_s1 (overfits; interesting negative result) and parseq_small_s1 (didn't fully converge) underperform.

Multi-seed analysis for headline:
- Aspect-preserved (n=3): mean T1 **12.68 ± 0.05** / T2 **21.88 ± 0.13**
- Published-resize VGG (n=3): mean T1 **13.15 ± 0.06** / T2 **22.87 ± 0.17**
- Both significantly beat 16.41 / 28.65 in absolute terms with 95% bootstrap CIs not overlapping the published values.

DiffOCR is the first competitive **non-autoregressive diffusion** recognizer on this benchmark, at 10.0M params, beating SVTR-Small (3.91M) on Test-II by 0.57 CER (within CI uncertainty). Its iterative-refinement decoding is fully parallel, with no autoregression, which is the main appeal vs. PARSeq/TrOCR-style AR decoders.

## 2026-06-06: Inference module and Gradio web demo

Inference + web app:
- `src/infer.py`: `HHDRecognizer` class loading any saved checkpoint (CRNN / SVTR / PARSeq / DiffOCR / TrOCR-when-fixed) and exposing `.predict(image_path | PIL | ndarray)` + a CLI (`python -m src.infer --ckpt … --image …`). Smoke-tested CLI against Test-I[0] → exact-match prediction.
- `app/web.py`: Gradio web app: pick checkpoint, pick bundled HHD-Ethiopic test sample OR upload an image; outputs predicted Ethiopic text + colour-coded diff against ground truth + CER + model metadata. The strongest aspect-preserved seed is the default. Smoke-tested with HTTP probe against the launched server (gradio 5.50 + gradio_client 1.14; gradio 4.44 had a `additionalProperties: True` schema bug that broke launch).
- `slurm/web_demo.sbatch`: SLURM script that allocates 1 L4 GPU and launches the demo, prints port-forwarding command for laptop access.

requirements.txt re-frozen with gradio 5.50 / huggingface_hub 0.36.

**All result artefacts ready**:
- `tables/main_results.{csv,tex,md}`: final table including DiffOCR
- `figures/comparison_bars/comparison_bars.{png,pdf}`: grouped bar chart vs published + ours
- `figures/training_curves/{loss_and_val_cer,test_cer}.{png,pdf}`: overlay of all training trajectories
- `figures/confusion/*_top30.{png,pdf}`: diacritic-confusion top-30 bar charts per run/split
- `figures/eda/{image_dimensions,char_length,char_frequency_top50}.{png,pdf}`: EDA figs
- `figures/eda_samples.{png,pdf}`: sample text-line images per split
- `results/crnn_vgg_s1/robustness_test_{I,II}.{csv,png,pdf}`: corruption-curve sweep
- `tables/confusion_*_top30.csv`: confusion matrices
- `data_card.md`: dataset summary

eval_all (job 33950351) plowing through every completed checkpoint, ~37 min/run (most time spent in `bootstrap_ci`). After it completes, every model will have predictions CSV, metrics.json with CIs, and diacritic-confusion CSV.

---

## 2026-06-08: Consistency fixes (no new training)

- Standard deviations across seeds are now sample standard deviations throughout, which gives
  Test-II **21.88 ± 0.15** for the aspect-preserving CRNN-VGG (the ± 0.13 above mixed the
  population and sample formulas).
- The WER word splitter now covers the whole Ethiopic punctuation block, U+1360 to U+1368
  including U+1367; that character does not occur in the test sets, so WER is unchanged.
- The DiffOCR `inference_steps` default is 12, matching the evaluation during training.
- The PARSeq model is a plain causal autoregressive decoder: the permuted-training path in
  `src/models/parseq.py` is never invoked.

---

## 2026-09-07: Corrections and new analyses (no new training)

New measurements made for the final write-up overturned part of the earlier interpretation.
Every correction below was confirmed against the source files.

1. **HPopt-Attn-CTC was trained for 100 epochs, not 25.** Belay et al. state in their appendix
   that the attention-CTC results of their main paper come from 100-epoch training. The 25-epoch
   figure applies only to their four mmocr scene-text baselines, and the plain-CTC pair used 15.
   Our models used 80 epochs, a shorter schedule than the bound they beat, so a longer schedule
   cannot explain the gain.
2. **The schedule contributes almost nothing for the CRNN family.** From
   `checkpoints/*/metrics.jsonl`, continuing from epoch 25 to epoch 80 changes Test-II CER by
   -0.23 points on average over nine runs and worsens it in four of the nine, inside the
   0.31-point seed range. Every CRNN run crosses the published 28.65 bound within 3 to 6 epochs.
   SVTR (-3.65) and DiffOCR are the opposite case and do need the longer schedule.
3. **`crnn_vgg_s2` was first reported from an epoch-12 snapshot.** Its original
   `results/crnn_vgg_s2/metrics.json` (13.87 / 23.46) matched epoch 12 of its own training log to
   three decimals, while the run completed 80 epochs at 13.16 / 22.79. It was re-evaluated from
   its best checkpoint into `results/crnn_vgg_s2_reeval/` (**13.16 / 22.79**), and
   `tables/main_results.csv` was fixed. The stale snapshot is not part of this repository.
4. **PARSeq ran 5 of 80 epochs, not 80.** `checkpoints/parseq_small_s1/metrics.jsonl` has 5 rows,
   and `best.pt` is at epoch 4 with a validation CER of 40.3 that was still falling by more than
   20 points per epoch. Its 40.40 / 55.99 is therefore not a converged negative result, and the
   autoregressive design axis remains untested.
5. **The robustness sweep uses `crnn_vgg_s1`** (48×368, light augmentation, Test-II CER 22.77),
   not the best model; blur 0 in `robustness_test_II.csv` equals that run's CER exactly.
6. **The confusion analysis uses seed 2** of the aspect-preserving CRNN-VGG, the best of the
   three seeds on Test-II (21.73).
7. **The average-human figures** of Belay et al. come from their running text, not from their
   Table 3.

**The ConvNeXt result is not over-fitting but CTC time-step starvation.** `ConvNeXtBackbone` uses
timm stage 3 (`out_indices=(3,)`, stride 32), so a 48×512 input gives a 1×16 feature map and the
CTC head gets **T=16** time steps against **127** for VGG. Five checks confirm it: (a) the
measured feature shapes; (b) ConvNeXt's longest prediction anywhere in the 22,310 test lines is
exactly 16 characters, against 44 for VGG; (c) its per-length CER tracks VGG up to the 13–16
bin and then rises steeply (14.26 → 29.82 → 51.06 → 60.71 → 72.93 on Test-I); (d) 64.0% of its
Test-II edits are deletions, against 21.2% for VGG seed 1; (e) re-weighting the Test-II
per-length CER to the Test-I length mixture shrinks its gap between the splits from 14.21 to 8.51
points, while the same re-weighting leaves the VGG gap essentially unchanged (9.02 → 9.37).
`nn.CTCLoss(zero_infinity=True)` was also verified to return a loss **and a gradient** of exactly
0.0 when a target is longer than T, so these lines contributed nothing to training (25.4% of the
training lines once repeated characters are counted; this entry first gave 25.2%). A suggested
second defect, invisible bottom input rows, did not reproduce: a gradient test shows 0 of 48 dead
input rows.

**Error modes** (seed 2, Test-II, `scripts/error_mode_breakdown.py`): 61,049 edits, of which
42,602 are substitutions (69.8%), 13,560 deletions (22.2%) and 4,887 insertions (8.0%).
Substitutions split into 41.9% vowel-order (diacritic), 46.7% base-consonant and 11.5%
non-syllabic, so diacritics are not the single dominant error mode. Vowel-order errors are highly
concentrated: the pair of the 1st and 6th orders alone accounts for 52.5% of them. 48.9% of the
deletions are the word space ፡ and the full stop ።.

**Compute** (`scripts/compute_cost.py`): parameter counts are exact, and training time is the sum
of `epoch_time_s` over 80 epochs (CRNN-VGG aspect-preserving 1.67 h, ConvNeXt 1.66 h, DiffOCR
4.90 h, SVTR 5.62 h). CPU latency is measured from the trained checkpoints. The MAC counts from
thop are logged but not used, because thop has no rule for `nn.MultiheadAttention` and misses
hand-written attention products, so they are complete only for the CRNN models.

**New or rewritten scripts:** `scripts/{figure_length_cer,error_mode_breakdown,compute_cost,figure_training_curves}.py`;
`scripts/figure_comparison_bars.py` was rewritten so that every run has its own label and colour.
New tables: `tables/{length_cer,error_modes,compute_cost}.csv`.

---

## 2026-11-02: Further analyses (no new training)

Every new number comes from the existing checkpoints and prediction files. The scripts are in
`scripts/analysis/` and their outputs in `results/analysis/`.

- **CER definition verified.** Eq. 3 of Belay et al. defines a corpus-level CER. Their released
  `Test_HPopt-Attn-CTC.ipynb` computes it in the same way and prints 16.397 (published 16.41).
  Our ground-truth transcriptions equal the benchmark's `y_test_*.npy` label arrays on all 22,310
  test lines (no mismatch; the Test-II denominator is 280,917 characters in both), and the
  notebook procedure reproduces the CER of all 12 of our runs to within 0.001. The mean of
  per-line CERs for the best configuration over three seeds is 13.60 on Test-I and 22.35 on
  Test-II.
- **Corrections.** (1) The released HPopt-Attn-CTC notebooks letterbox each line (white padding
  to 368:48, then a bilinear resize), so our 48×368 runs are a stretch control (`INTER_AREA`)
  that matches the published input size but not the published resizing rule. (2) The 512:48
  canvas is the 64th percentile of Test-II aspect ratios, not the 95th: 35.8% of Test-II and
  30.9% of training lines are compressed horizontally (median factors 1.38 and 1.20) and never
  cropped (`_pad_to_aspect`). (3) The benchmark's released mmocr CRNN configuration trains on
  32×100 grayscale stretches (MiniVGG, 26 frames, so 6.7% of the training lines cannot be aligned
  under `zero_infinity`) and tests at 32×176 with Adadelta at a constant learning rate of 1.0, so
  it does not share our geometry. (4) `abl_aug_none_s1` (run A) and `abl_pub_48x368_s1` (run B)
  have identical configurations and seeds, and their labels had been swapped between the results
  and schedule tables. (5) The augmentation levels are nested, and shear is measured in degrees.
  (6) The published values carry no confidence intervals, so "non-overlapping intervals" in
  earlier entries should read: every seed's 95% interval, line-level or page-level, excludes the
  published values. (7) The 6.77-point Test-II gain over HPopt-Attn-CTC decomposes into 3.96
  points already present in the least-modified configuration (48×368 stretch, no augmentation,
  25 epochs), 0.59 from training it to 80 epochs, 1.23 from light augmentation and 0.99 from the
  aspect-preserving canvas with medium augmentation, at least 0.58 of which is geometry. This
  supersedes the "roughly half" estimate of 2026-06-04.
- **Page-level bootstrap.** Test-II is stored in source order (lag-1 autocorrelation of line
  height 0.85, against 0.00 for Test-I), so it is resampled with a circular moving-block
  bootstrap (2,000 resamples). For the three-seed mean on Test-II, line-level resampling gives
  [21.65, 22.09], 50-line blocks give [21.03, 22.74] and 800-line blocks give [19.00, 24.65].
  Paired ablation differences barely move.
- **Character language model.** A character 6-gram with interpolated Kneser-Ney smoothing (order
  chosen by validation perplexity, 13.93 at order 6) is combined with the CTC posteriors by prefix
  beam search (beam 10, alpha 0.4, beta 1.0, tuned on the seed-2 validation data,
  12.707 → 12.099). Over three seeds: Test-I 12.68 → 12.22 [-0.53, -0.39]; Test-II
  21.87 → 20.24 [-1.72, -1.54]; Test-II WER 56.91 → 51.66. Seed 2, Test-II: base-consonant
  substitutions -13.9%, vowel-form -7.2%, deletions -1.3%, insertions +3.7%; word-space deletions
  -10.7%, full stops +2.3%.
- **Convergence.** Test-II change over the last 20 epochs: CRNN runs -0.19 to +0.12, SVTR -0.35,
  DiffOCR -1.96 (26.56 → 24.60).
- **DiffOCR steps and latency.** One CPU session (Intel Core i7-14700, 4 threads, batch 32):
  CRNN-VGG 56 ms per line; DiffOCR with K = 1/2/4/8/12/16 refinement steps takes
  52/59/72/96/113/137 ms, with Test-II CER 31.18/30.43/27.99/25.39/24.60/24.18 and Test-I CER
  18.65/18.21/16.62/14.85/14.40/14.19 (K = 12 reproduces the GPU values). A second timing session
  overlapped other jobs and is not used.
- **Figures.** The summary figures were redrawn at print size (6.5 to 8 pt text, one colour per
  model across all figures) into `figures/analysis/`.

**Still open:** a 48×512 run with light augmentation (about 1.7 GPU-hours) would remove the
remaining confound between geometry and augmentation.
