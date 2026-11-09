#!/bin/bash
#SBATCH --job-name=eval_vgg_s1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/eval_vgg_%j.out
#SBATCH --error=slurm_logs/eval_vgg_%j.err

source slurm/_common.sh

CKPT=checkpoints/crnn_vgg_s1/best.pt
OUT=results/crnn_vgg_s1
mkdir -p "$OUT"

echo "=== eval ${CKPT} ==="
python -u -m src.eval --ckpt "$CKPT" --output_dir "$OUT" --bootstrap 1000 --batch_size 128

echo "=== diacritic confusion ==="
python -u scripts/diacritic_confusion.py --predictions "$OUT/predictions_test_I.csv"  --top_k 30
python -u scripts/diacritic_confusion.py --predictions "$OUT/predictions_test_II.csv" --top_k 30

echo "=== robustness sweep (Test-II) ==="
python -u scripts/robustness_sweep.py --ckpt "$CKPT" --out_dir "$OUT" --split test_II --batch_size 128

echo "=== qualitative gallery ==="
python -u scripts/qualitative_gallery.py --predictions "$OUT/predictions_test_I.csv"  --n_samples 12
python -u scripts/qualitative_gallery.py --predictions "$OUT/predictions_test_II.csv" --n_samples 12

echo "End: $(date -Iseconds)"
