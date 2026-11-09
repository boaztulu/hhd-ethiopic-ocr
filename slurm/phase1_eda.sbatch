#!/bin/bash
#SBATCH --job-name=hhd_eda
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/eda_%j.out
#SBATCH --error=slurm_logs/eda_%j.err

source slurm/_common.sh

echo "==> Phase 1 EDA (verification + plots)"
python -u scripts/phase1_eda.py
echo "End: $(date -Iseconds)"
