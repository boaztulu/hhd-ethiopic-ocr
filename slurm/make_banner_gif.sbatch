#!/bin/bash
#SBATCH --job-name=banner_gif
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/banner_%j.out
#SBATCH --error=slurm_logs/banner_%j.err

source slurm/_common.sh
python -u app/make_banner_gif.py
echo "End: $(date -Iseconds)"
