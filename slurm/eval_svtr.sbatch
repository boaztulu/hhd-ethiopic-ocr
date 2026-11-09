#!/bin/bash
#SBATCH --job-name=eval_svtr
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/eval_svtr_%j.out
#SBATCH --error=slurm_logs/eval_svtr_%j.err

source slurm/_common.sh

CKPT=${CKPT:-checkpoints/svtr_small_s1/best.pt}
OUT=${OUT:-results/svtr_small_s1}
mkdir -p "$OUT"

echo "=== eval ${CKPT} (batch=32) ==="
python -u -m src.eval --ckpt "$CKPT" --output_dir "$OUT" --bootstrap 1000 --batch_size 32

echo "=== diacritic confusion ==="
python -u scripts/diacritic_confusion.py --predictions "$OUT/predictions_test_I.csv"  --top_k 30 || true
python -u scripts/diacritic_confusion.py --predictions "$OUT/predictions_test_II.csv" --top_k 30 || true

echo "End: $(date -Iseconds)"
