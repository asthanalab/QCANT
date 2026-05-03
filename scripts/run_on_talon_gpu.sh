#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  cat <<'EOF'
Usage: scripts/run_on_talon_gpu.sh '<command>'

Environment overrides:
  PARTITION  Slurm partition (default: talon-gpu32)
  NODE       Optional node name (e.g. talon33)
  GPUS       GPUs per task (default: 1)
  CPUS       CPUs per task (default: 8)
  MEM        Memory request (default: 64G)
  TIME       Walltime (default: 04:00:00)

Example:
  PARTITION=gpu-code-test NODE=talon33 GPUS=1 \
    scripts/run_on_talon_gpu.sh 'python scripts/benchmark_gpu_speedups.py --gpu-device lightning.gpu'
EOF
  exit 1
fi

CMD="$*"
PARTITION="${PARTITION:-talon-gpu32}"
NODE="${NODE:-}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-04:00:00}"

SRUN_ARGS=(
  -p "$PARTITION"
  -N1
  -n1
  --gpus="$GPUS"
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --time="$TIME"
)

if [ -n "$NODE" ]; then
  SRUN_ARGS+=(--nodelist "$NODE")
fi

exec srun "${SRUN_ARGS[@]}" bash -lc "$CMD"
