# Analyses

Scripts behind the later analyses recorded in `RUN_LOG.md` (2026-11-02). All of them run on a CPU
with the project environment (`python -P <script>`). Outputs go to `results/analysis/` and
figures to `figures/analysis/`. Test images are read from the distributed zip files in
`HHD-Ethiopic-data/` (`train_raw.zip`, `test.zip`), which is much faster than reading the
extracted image folders one file at a time from a cloud-synced disk.

Run them in this order, since each one's inputs come from the step before.

| Script | What it produces |
|---|---|
| `geometry.py` | Line width, height and aspect ratio per split; share of lines wider than the 368:48 and 512:48 canvases and their compression; letterbox ink-band height; share of lines that CTC cannot align at 16, 26, 45, 91 and 127 frames; storage-order autocorrelation |
| `rescore.py` | Checks the ground truth against the benchmark's label arrays and re-scores every run with the CER procedure of the benchmark's test notebook, plus the mean of per-line CERs and NED; saves per-line edit distances |
| `logs.py` | Best-validation epoch against last epoch, late-schedule slopes and the first epoch below 28.65, from `checkpoints/*/metrics.jsonl` |
| `bootstrap.py` | Line-level and moving-block (page-scale) bootstrap intervals and paired contrasts |
| `aspect_strat.py` | Ablation gains stratified by aspect ratio |
| `cpu_jobs.py` | CTC posteriors of the three aspect-preserving CRNN-VGG seeds (validation, Test-I, Test-II), DiffOCR accuracy at 1 to 16 refinement steps and the CPU latency of all models in one session; needs the trained weights |
| `lm_decode.py` | Character n-gram language model (interpolated Kneser-Ney) with CTC prefix beam search; `ppl` mode chooses the order and `tune` mode chooses alpha and beta on the validation data |
| `lm_eval.py` | Greedy, beam and beam-plus-LM evaluation on both test sets, punctuation deletions and paired intervals (`python -P lm_eval.py 0.4 1.0`) |
| `lm_errors.py` | Which error types the language model removes (seed 2) and how uncertain the recognizer is on each split |
| `summary_figures.py` | Summary figures at print size (12.2 cm wide) |

The CTC posterior arrays (`post_*.npz`, about 450 MB) are not included; `python -P cpu_jobs.py ctc`
regenerates them from the trained weights in about 40 minutes on a 20-core CPU. The reported
latencies come from `latency_session1.json`, which was timed on an otherwise idle machine.
