#!/bin/bash
#SBATCH --job-name=eval_one
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/eval_one_%j.out
#SBATCH --error=slurm_logs/eval_one_%j.err

source slurm/_common.sh

# Required: RUN=<run name>. Optional: BS=<batch>, BOOT=<bootstrap>.
RUN=${RUN:?need RUN=<checkpoint subdir>}
BS=${BS:-128}
BOOT=${BOOT:-200}

CKPT=checkpoints/$RUN/best.pt
OUT=results/$RUN
mkdir -p "$OUT"

echo "=== eval $RUN (batch=$BS bootstrap=$BOOT) ==="
python -u -m src.eval --ckpt "$CKPT" --output_dir "$OUT" --bootstrap "$BOOT" --batch_size "$BS"

echo "=== diacritic confusion ==="
python -u scripts/diacritic_confusion.py --predictions "$OUT/predictions_test_I.csv"  --top_k 30 || true
python -u scripts/diacritic_confusion.py --predictions "$OUT/predictions_test_II.csv" --top_k 30 || true

echo "End: $(date -Iseconds)"
