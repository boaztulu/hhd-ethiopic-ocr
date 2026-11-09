#!/bin/bash
# Submit the augmentation + preprocessing ablations after the first wave clears.
set -euo pipefail
cd "$(dirname "$0")/.."

# Preprocessing ablation (48x368 stretch, no aug; same settings as abl_aug_none)
sbatch --export=ALL,CONFIG=configs/crnn_vgg_published.yaml,OUT_DIR=checkpoints/abl_pub_48x368_s1,SEED=1 \
       --job-name=abl_pub_48x368 slurm/train_ctc.sbatch

# Preprocessing ablation (48x512 aspect-preserved, medium aug)
sbatch --export=ALL,CONFIG=configs/crnn_vgg_aspect.yaml,OUT_DIR=checkpoints/abl_asp_48x512_s1,SEED=1 \
       --job-name=abl_asp_48x512 slurm/train_ctc.sbatch

# Augmentation level ablations
sbatch --export=ALL,CONFIG=configs/crnn_vgg_aug_none.yaml,OUT_DIR=checkpoints/abl_aug_none_s1,SEED=1 \
       --job-name=abl_aug_none slurm/train_ctc.sbatch

sbatch --export=ALL,CONFIG=configs/crnn_vgg_aug_heavy.yaml,OUT_DIR=checkpoints/abl_aug_heavy_s1,SEED=1 \
       --job-name=abl_aug_heavy slurm/train_ctc.sbatch

echo "Ablation jobs queued."
