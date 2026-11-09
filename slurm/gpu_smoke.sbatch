#!/bin/bash
#SBATCH --job-name=hhd_smoke
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/smoke_%j.out
#SBATCH --error=slurm_logs/smoke_%j.err

source slurm/_common.sh

echo "==> GPU smoke (CPU smoke already passed)"
python -u scripts/smoke_test.py

echo "==> CRNN-vgg, 100 train steps"
python -u -m src.train_ctc \
    --config configs/crnn_vgg.yaml \
    --output_dir checkpoints/_smoke_crnn_vgg \
    --seed 1 \
    --device cuda \
    --max_steps 100 \
    --no_eval_test

echo "==> SVTR-small, 100 train steps"
python -u -m src.train_ctc \
    --config configs/svtr_small.yaml \
    --output_dir checkpoints/_smoke_svtr \
    --seed 1 \
    --device cuda \
    --max_steps 100 \
    --no_eval_test

echo "End: $(date -Iseconds)"
