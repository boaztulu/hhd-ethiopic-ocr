#!/bin/bash
#SBATCH --job-name=hhd_eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/eval_%j.out
#SBATCH --error=slurm_logs/eval_%j.err

source slurm/_common.sh

# Required env: CKPT=path/to/best.pt, OUT=results/<run>
CKPT=${CKPT:?need CKPT=path/to/best.pt}
OUT=${OUT:?need OUT=results/<run>}
BOOT=${BOOT:-1000}

echo "Evaluating $CKPT  -> $OUT"
python -u -m src.eval --ckpt "$CKPT" --output_dir "$OUT" --bootstrap "$BOOT" --batch_size 128
echo "Diacritic confusion analysis (Test-I, Test-II)..."
python -u scripts/diacritic_confusion.py --predictions "$OUT/predictions_test_I.csv"  --top_k 30 || true
python -u scripts/diacritic_confusion.py --predictions "$OUT/predictions_test_II.csv" --top_k 30 || true
echo "End: $(date -Iseconds)"
