#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m morphmimic.scripts.train_human \
  --teacher-root "$root/runs/releases/shared-v2-contact-30000" \
  --data-root "$root/prepared/human-v1" \
  --run-root "$root/runs/human-adaptation" --run-name human-v1 \
  --gpu-ids 0 2 3 --num-envs 2048 --updates 10000 "$@"
