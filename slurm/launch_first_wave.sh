#!/bin/bash
# Submit the first wave of training jobs (one seed each, 4 models).
# Usage: bash slurm/launch_first_wave.sh [seed]
# Each job auto-resumes from <out>/last.pt on requeue.

set -euo pipefail
SEED=${1:-1}
cd "$(dirname "$0")/.."

# CRNN-VGG, 48x368 stretch, light augmentation
sbatch \
    --export=ALL,CONFIG=configs/crnn_vgg.yaml,OUT_DIR=checkpoints/crnn_vgg_s${SEED},SEED=${SEED} \
    --job-name=crnn_vgg_s${SEED} \
    slurm/train_ctc.sbatch

# CRNN-ConvNeXt (modern CTC)
sbatch \
    --export=ALL,CONFIG=configs/crnn_convnext.yaml,OUT_DIR=checkpoints/crnn_convnext_s${SEED},SEED=${SEED} \
    --job-name=crnn_convnext_s${SEED} \
    slurm/train_ctc.sbatch

# SVTR-small CTC
sbatch \
    --export=ALL,CONFIG=configs/svtr_small.yaml,OUT_DIR=checkpoints/svtr_small_s${SEED},SEED=${SEED} \
    --job-name=svtr_small_s${SEED} \
    slurm/train_ctc.sbatch

# PARSeq (AR)
sbatch \
    --export=ALL,CONFIG=configs/parseq_small.yaml,OUT_DIR=checkpoints/parseq_small_s${SEED},SEED=${SEED} \
    --job-name=parseq_small_s${SEED} \
    slurm/train_ar.sbatch

echo "Submitted seed=${SEED}. Check with: squeue -u $USER"
