#!/bin/bash
#SBATCH --job-name=hhd_ctc
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --output=slurm_logs/ctc_%j.out
#SBATCH --error=slurm_logs/ctc_%j.err

source slurm/_common.sh

CONFIG=${CONFIG:-configs/crnn_convnext.yaml}
OUT_DIR=${OUT_DIR:-checkpoints/crnn_convnext_s${SEED:-1}}
SEED=${SEED:-1}
RESUME_ARG=""
INIT_ARG=""
if [ -n "${RESUME:-}" ]; then RESUME_ARG="--resume ${RESUME}"; fi
if [ -n "${INIT_FROM:-}" ]; then INIT_ARG="--init_from ${INIT_FROM}"; fi
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "Config:    $CONFIG"
echo "OutDir:    $OUT_DIR"
echo "Seed:      $SEED"
echo "Resume:    ${RESUME:-<none>}"
echo "Init from: ${INIT_FROM:-<none>}"
echo "Extra:     $EXTRA_ARGS"

mkdir -p "$OUT_DIR"

# Auto-resume if last.pt exists and no explicit resume given
if [ -z "${RESUME:-}" ] && [ -f "$OUT_DIR/last.pt" ]; then
    RESUME_ARG="--resume $OUT_DIR/last.pt"
    echo "[auto-resume] using $OUT_DIR/last.pt"
fi

trap 'echo "[SLURM] Forwarding SIGUSR1"; kill -USR1 $PID 2>/dev/null || true' USR1

python -u -m src.train_ctc \
    --config "$CONFIG" \
    --output_dir "$OUT_DIR" \
    --seed "$SEED" \
    --device cuda \
    $RESUME_ARG $INIT_ARG $EXTRA_ARGS &
PID=$!
wait $PID
STATUS=$?
echo "Train exited with status $STATUS"
echo "End: $(date -Iseconds)"
exit $STATUS
