#!/bin/bash
#SBATCH --job-name=hhd_robust
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/robust_%j.out
#SBATCH --error=slurm_logs/robust_%j.err

source slurm/_common.sh

CKPT=${CKPT:?need CKPT=path/to/best.pt}
OUT=${OUT:?need OUT=results/<run>}

python -u scripts/robustness_sweep.py --ckpt "$CKPT" --out_dir "$OUT" --split test_II --batch_size 128
python -u scripts/robustness_sweep.py --ckpt "$CKPT" --out_dir "$OUT" --split test_I  --batch_size 128
echo "End: $(date -Iseconds)"
