#!/bin/bash
# Submit seed=2 and seed=3 runs of the leading model (CRNN-vgg variant).
# Adjust CONFIG via env if a different model wins after first wave finishes.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/crnn_vgg.yaml}
PREFIX=${PREFIX:-crnn_vgg}

for SEED in 2 3; do
    sbatch \
        --export=ALL,CONFIG=${CONFIG},OUT_DIR=checkpoints/${PREFIX}_s${SEED},SEED=${SEED} \
        --job-name=${PREFIX}_s${SEED} \
        slurm/train_ctc.sbatch
done
echo "Queued seeds 2 and 3 for ${PREFIX}."
