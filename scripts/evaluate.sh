#!/usr/bin/env bash
set -euo pipefail
iteration=${1:?Usage: evaluate.sh CHECKPOINT_ITERATION}
[[ "$iteration" =~ ^[0-9]+$ ]] || { echo 'Checkpoint iteration must be numeric' >&2; exit 2; }
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
for robot in q1 bumi g1; do
  for subset in val lafan1-all; do
    python -u -m morphmimic.scripts.eval \
      --ckpt "$root/runs/shared-v1/$robot/ckpt_$iteration.pt" \
      --subset-yaml "$root/prepared/full-v1/$robot/$subset.yaml" \
      --num-envs 256 --library-dtype float16 --gpu 0 \
      --out "$root/reports/eval-$iteration/$robot-$subset.csv"
  done
done
