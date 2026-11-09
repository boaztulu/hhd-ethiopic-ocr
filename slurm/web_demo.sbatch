#!/bin/bash
#SBATCH --job-name=hhd_demo
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/demo_%j.out
#SBATCH --error=slurm_logs/demo_%j.err

source slurm/_common.sh

PORT=${PORT:-7860}
DEVICE=${DEVICE:-cuda}

NODE=$(hostname -s)
echo "Starting Gradio web demo at http://${NODE}:${PORT}"
echo "From your laptop, port-forward with:"
echo "   ssh -L ${PORT}:${NODE}:${PORT} ${USER}@hpg.rc.ufl.edu"
echo "then open  http://localhost:${PORT}  in your browser."

python -u -m app.web --host 0.0.0.0 --port "$PORT" --device "$DEVICE"
