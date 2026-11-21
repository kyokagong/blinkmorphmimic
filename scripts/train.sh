#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
data=${BLINKMORPH_PREPARED:-$root/prepared/full-v1}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m morphmimic.scripts.train \
  --robots q1 bumi g1 --share all --gpu-ids 0 1 2 \
  --subset-yamls "$data/q1/train.yaml" "$data/bumi/train.yaml" "$data/g1/train.yaml" \
  --latent-align-weight 0 --library-dtype float16 --optimizer adam --policy-ema .995 \
  --num-envs 2048 --replay-capacity 256 --wm-batch 1024 --wm-batches 5 \
  --imag-batch 1024 --wm-hidden 1024 1024 1024 --policy-hidden 2048 1024 512 256 \
  --iterations 30000 --wm-only-iters 500 --ckpt-every 1000 --log-every 50 \
  --log-dir "$root/runs" --run-name shared-v1 "$@"
