# BlinkMorphMimic

The live preview now uses the EMA checkpoint from the **random-initialized 0→5000 optimization experiment**, with synchronized Q1/Bumi/G1 at **1280×720**. Both optimization and Adam baseline runs used **2048 environments per GPU**, without earlier learned checkpoints. See [current results, curves and measured limits](docs/random-5000-results.md). Earlier 30,000-iteration and 1080p records remain historical evidence.

The single-user preview keeps independent NVENC encoding; multi-viewer shared encoding was cancelled.

Shared morphology-conditioned world model and tracking policy for Q1 (22 native joints), Bumi V3 (21), and Unitree G1 (29). Built from the pinned official MorphMimic core with sparse29-joint mappings, explicit154-dimensional morphology/mask descriptors, policyEMA, optionalMuon, source-level data splits, and actual-policy online simulation.

Execution and limitations: [documentation archive](docs/README.md), [actual implementation](docs/implementation.md), [training/evaluation](docs/training-evaluation.md), [runtime/recovery](docs/runtime.md).

## Runtime

Use the Dockerfile with the locally available frozen BlinkMimic dependency image. The verified runtime is Python3.12, PyTorch2.11+cu130, mjlab1.6.0, MuJoCo/MuJoCoWarp3.11 and Warp1.17. Q1 EPA horizon patch is inherited from that dependency image; original collisions and solver remain enabled. No IsaacLab import is needed by this project.

```bash
docker build -t blinkmorphmimic:local .
```

Source module namespace remains `morphmimic` for compatibility. Supply `PYTHONPATH=$PWD/src` for direct Python execution. Original datasets/assets stay under `/data/kk/data`, mounted read-only.

## Fresh optimized training

The current entry is `python -m scripts.train_optimized`: three robots on three GPUs,
2048 environments per GPU, 5000 training iterations, Muon, and no initialization
checkpoint by default. Motion and authentic paired human interfaces share the policy.
The script keeps the world-model readiness and native physics overflow checks enabled.
This preset does not enable the separately developed optional controller-aware mode.

Use a completed composite prepared corpus with `manifest.json`, each robot's
`train.yaml`, authentic human sidecars, and the matching `contact-floor.json` receipt.
Do not rerun conversion for an already verified corpus. See
[training/evaluation](docs/training-evaluation.md) for data preparation and contracts.
The manifest's absolute motion/human paths must remain accessible inside the container;
`/data/kk/data` and the existing prepared corpus are mounted read-only below.

From this source checkout, build the runtime as above, then set the existing composite
path and a new output root/name. These commands first print the full configuration,
freeze the exact source/config/data/image identity, then launch that same configuration:

```bash
set -euo pipefail
prepared=/absolute/path/to/composite
artifacts=/absolute/path/to/new-training-output
run_name=fresh-muon-2048-5000
mkdir -p "$artifacts/receipts"
image_id=$(docker image inspect --format '{{.Id}}' blinkmorphmimic:local)
receipt="$artifacts/receipts/$run_name.json"
config="$artifacts/receipts/$run_name-config.json"

runtime=(docker run --rm --network host --shm-size=16g --gpus all
  -e PYTHONPATH=/work/src:/work -e BLINKMORPH_RUNTIME_IMAGE_ID="$image_id"
  -v "$PWD:/work:ro" -v "$prepared:$prepared:ro" -v "$artifacts:$artifacts:rw"
  -v /data/kk/data:/data/kk/data:ro
  -v /data/kk/blinkmorphmimic/prepared:/data/kk/blinkmorphmimic/prepared:ro
  --workdir /work "$image_id")
args=(--data-root "$prepared" --log-dir "$artifacts" --run-name "$run_name"
  --gpu-ids 0 2 3 --freeze-receipt "$receipt")

# New names only; neither a configuration nor a receipt is overwritten.
test ! -e "$config" && test ! -e "$receipt" && test ! -e "$artifacts/$run_name"
(set -o noclobber; "${runtime[@]}" python -m scripts.train_optimized "${args[@]}" --dry-run > "$config")
"${runtime[@]}" python -m scripts.freeze_optimized \
  --data-manifest "$prepared/manifest.json" --source-root /work \
  --config "$config" --image-id "$image_id" --output "$receipt"
"${runtime[@]}" python -m scripts.train_optimized "${args[@]}"
```

Keep the source, config, immutable image ID and receipt unchanged after freezing.
`optimization_schema=1` uses the prepared manifest; the supplied freeze receipt and
`BLINKMORPH_RUNTIME_IMAGE_ID` additionally bind the exact source/config/runtime at
startup and resume. No `--initial-checkpoint-root` or `--resume` is used above.
The configuration dry run performs no training; receipt creation performs no GPU work.
Set the three GPU IDs to the actual available devices before freezing.

Evaluation, its watcher and plotting default to checkpoints 0, 1000, …, 5000. For historical
runs, explicitly supply `--iterations 0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000`;
checkpoint, manifest and artifact checks remain enforced. The train-only probe defaults
to `--training-target 5000`; use `--training-target 10000` explicitly for a historical
10000-iteration run. Its budget ledger binds that target. An old ledger without a target
is rejected: preserve it and choose a separate output directory instead of reinterpreting it.
A missing ledger is also rejected when prior output artifacts exist; preserve those artifacts
and restore the original ledger instead of resetting spent time. Only an empty output
directory (apart from its worker lock) can start a new ledger. The editable JSON ledger
is a local accounting record, not protection against arbitrary malicious edits to spent time;
that would require a separate trusted record.
The probe exits when all final checkpoints exist and no work fits its remaining budget.

The legacy `bash scripts/train.sh` entry remains available for historical Adam/30000
workflows and their existing callers. It is not the fresh optimized preset above.
Earlier `shared-v1`/`shared-v2-contact` training and 30000-step preview measurements
remain historical evidence; these default changes do not alter or retrain any saved run.
Policies remain simulation policies, without hardware closed-loop qualification.

The completed experiment jointly trains motion and authentic human five-point interfaces using BONES-SEED/LAFAN1; MotionDecode is motion-only because matching human sources were not found. Muon is active for hidden matrices of the world model and both policy interfaces; excluded layers and critics use Adam. Native actuator delay is enabled in training. Policies are not qualified for hardware.

Optional modes and their separate qualification limits: [native controller model](docs/controller-aware-native-v1.md), [single-GPU multi-robot training](docs/single-gpu-training.md). Neither mode was used by the completed random 5000/2048 comparison.

## Preview

See [preview instructions](preview/README.md). Only fixed checkpoints may be selected. The server uses the same trained policy, history, observations, native action mapping and physics environment as training; reference ghost and actual policy are distinct. The public simulation defaults to the fixed random-initialized 5000-iteration EMA release: [open preview](http://1.94.0.106:31769/). All 252 matched checkpoint evaluations are complete; [results and downloadable curves](docs/random-5000-results.md) include both improvements and remaining drift. Earlier releases stay selectable. See the [historical 15k–30k chart](docs/training-milestones-15000-30000.md) for that separate experiment.

## Attribution

See [third-party notices](THIRD_PARTY_NOTICES.md). Code licensing does not replace robot asset or motion dataset terms. No Git commit is needed by the workflow.
