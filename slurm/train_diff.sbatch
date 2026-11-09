#!/bin/bash
#SBATCH --job-name=diff_ocr
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/diff_%j.out
#SBATCH --error=slurm_logs/diff_%j.err

source slurm/_common.sh

CONFIG=${CONFIG:-configs/diff_ocr.yaml}
OUT_DIR=${OUT_DIR:-checkpoints/diff_ocr_s${SEED:-1}}
SEED=${SEED:-1}
RESUME_ARG=""
INIT_ARG=""
if [ -n "${RESUME:-}" ]; then RESUME_ARG="--resume ${RESUME}"; fi
if [ -n "${INIT_FROM:-}" ]; then INIT_ARG="--init_from ${INIT_FROM}"; fi

echo "Config: $CONFIG  OutDir: $OUT_DIR  Seed: $SEED"

mkdir -p "$OUT_DIR"
if [ -z "${RESUME:-}" ] && [ -f "$OUT_DIR/last.pt" ]; then
    RESUME_ARG="--resume $OUT_DIR/last.pt"
    echo "[auto-resume] $OUT_DIR/last.pt"
fi

trap 'echo "[SLURM] Forwarding SIGUSR1"; kill -USR1 $PID 2>/dev/null || true' USR1

python -u -m src.train_diff \
    --config "$CONFIG" \
    --output_dir "$OUT_DIR" \
    --seed "$SEED" \
    --device cuda \
    $RESUME_ARG $INIT_ARG &
PID=$!
wait $PID
STATUS=$?
echo "Exit: $STATUS  End: $(date -Iseconds)"
exit $STATUS
