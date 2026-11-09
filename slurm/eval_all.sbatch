#!/bin/bash
# Evaluate every `checkpoints/*/best.pt` and regenerate the master table.
#SBATCH --job-name=hhd_eval_all
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/eval_all_%j.out
#SBATCH --error=slurm_logs/eval_all_%j.err

source slurm/_common.sh

for CKPT in checkpoints/*/best.pt; do
    RUN=$(basename "$(dirname "$CKPT")")
    case "$RUN" in
        _smoke*) echo "skipping $RUN"; continue ;;
        abl_*|*_s1|*_s2|*_s3) ;;
        *) echo "skipping $RUN (unknown pattern)"; continue ;;
    esac
    OUT=results/$RUN
    if [ -f "$OUT/metrics.json" ] && [ "$OUT/metrics.json" -nt "$CKPT" ]; then
        echo "already eval'd: $RUN"
        continue
    fi
    # SVTR uses attention with O(W²) memory at W=128 → lower batch
    BS=128
    case "$RUN" in
        svtr_*|*_svtr_*) BS=32 ;;
        parseq_*|*_parseq_*) BS=64 ;;
    esac
    echo "=== eval $RUN  (batch=$BS) ==="
    python -u -m src.eval --ckpt "$CKPT" --output_dir "$OUT" --bootstrap 1000 --batch_size $BS || echo "[warn] eval failed for $RUN"
    python -u scripts/diacritic_confusion.py --predictions "$OUT/predictions_test_I.csv"  --top_k 30 || true
    python -u scripts/diacritic_confusion.py --predictions "$OUT/predictions_test_II.csv" --top_k 30 || true
done

echo "=== rebuilding master table + figures ==="
python -u scripts/build_main_table.py
python -u scripts/figure_comparison_bars.py
python -u scripts/plot_training_curves.py checkpoints/*/metrics.jsonl || true

echo "End: $(date -Iseconds)"
