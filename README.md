# BlinkMorphMimic

基于 MorphMimic 的多本体动作跟踪项目，面向 **Q1（22 关节）、Bumi（21 关节）、G1（29 关节）**，共享形态条件世界模型与跟踪网络，支持机器人 Motion 参考和真实配对人体五点输入。上游核心版本为 `770394b4a0cad9effbf807a6c03435d1699733d0`，来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

本地 `prepared/current.json` 指向修正后的原地起身混合数据和 20000 checkpoint 发布清单；实际在线版本需核对 `/api/status`。项目架构和开发入口见 [AGENTS.md](AGENTS.md)，正式设计与操作说明见 [docs/README.md](docs/README.md)。

## 对 MorphMimic 的具体优化

本项目保留上游形态条件学习框架，围绕三本体适配、训练效率、动作跟踪和可验证运行做了以下扩展与修正：

| 方向 | 具体实现 | 带来的能力与验证范围 |
|---|---|---|
| 多本体原生控制适配 | 统一 29 维 canonical 关节槽，形态描述增加缺失关节 mask；校正 Bumi 关节顺序、肘部符号和手部 frame，保留各本体原生 PD/物理参数 | 让 22/21/29 关节本体共享网络，避免缺失关节污染输入、损失和动作；范围为已适配的 Q1/Bumi/G1 |
| 优化器与稳定性 | WM、Motion/Human 内部矩阵使用 Muon，其余层及 critics 使用 Adam；策略 EMA、梯度等权共享及完整优化器恢复 | 改进组合训练配方的收敛表现；没有 Muon 单变量消融，不将全部增益归因于优化器 |
| GPU 训练执行 | 五阶段显式学习 CUDA graphs、collection graph、masked reset、异步 checkpoint；WM 就绪门槛控制 actor 开始学习 | 减少执行和保存开销，保留 reset、终止和物理溢出检查；组合配方吞吐改善见下表 |
| 世界模型与跟踪目标 | semi-implicit 关节动力学、reference reconstruction、接触一致性和足滑惩罚 | 在模型与奖励中加入动作、接触约束；完整配方有评价结果，逐项独立收益尚未证明 |
| 人体输入与数据契约 | 真实 SOMA/LAFAN1 配对人体五点，来源家族隔离划分、统一 50 Hz 和人体朝向校正 | 三本体可使用 pelvis/双踝/双腕五点跟踪；MotionDecode 保持 motion-only，避免伪造人体监督 |
| 起身与长尾动作训练 | 真实起身/翻转来源、来源分层探索 + 失败/几何误差重要性采样、统一自然垂手中立位和原地终态 | 提升部分恢复及长序列完成表现；严格原地起身和空翻仍有明显不足，不能把参考正确当成技能成功 |
| 数据与物理正确性 | Bumi MotionDecode SDK→XML 列重排、恢复原资产全身碰撞、地面校准与逐子步 sticky overflow 诊断 | 修复会直接破坏跟踪的输入和仿真问题；数据修复收益与训练算法收益分别报告 |
| 预览与可复现运行 | 固定 checkpoint/清单/镜像身份校验、真实物理三本体共享 actor、720p WebRTC/NVENC、可审计评价与恢复 | 展示真实策略行为并保留失败；播放性能不等于训练吞吐，也不代替技能或实机验收 |

另实现了 **native-delay 条件世界模型** 和 **单 GPU 顺序多本体训练** 两种可选模式，已做独立小规模运行/恢复验证；它们未参与随机 5000 对照，不能沿用该对照宣称性能收益。详见 [系统设计](docs/design.md)。

## 已验证的改善与限制

随机初始化 0→5000 的 OPT/Adam BASE 对照均为每本体 2048 环境，使用相同 held-out 数据及共同评价 reward；共完成 252 项固定 checkpoint 评价。最终 SEED 人体五点结果：

| 本体 | Adam BASE 回报 | OPT 回报 | 相对提高 | 失败片段 BASE→OPT /1072 |
|---|---:|---:|---:|---:|
| Q1 | 0.704228 | 0.735369 | 4.42% | 1→0 |
| Bumi | 0.657101 | 0.703389 | 7.04% | 2→2 |
| G1 | 0.610540 | 0.661187 | 8.30% | 12→8 |

相同稳态迭代窗口的观察吞吐为 **2.3370 对 1.65325 it/s，约提高 41.36%**。这是本项目内的单 seed、组合配方比较，actor 更新数、门槛和缓存历史不同，不能视为对官方 MorphMimic 的受控复现基准、冷启动加速或逐项优化收益。

后续 6000→7000 训练中，旧站稳判据下扩大起身成功数由 217/242 提高到 220/242，完整 dance1_subject1 完成数由 1/6 提高到 5/6；部分本体回退且人体长序列仍有漂移。修正原地参考后，7000→20000 的 18 个起身场景直立从 17/18 到 18/18，**严格原地保持 3 秒仅从 1/18 到 2/18**；两种起身标准不能直接混比。20000 评价中的实际整周翻转仍为 0/12。

Bumi MotionDecode 列顺序修复在固定 30000 模型下，将八极拳共同前 50 秒平均 XY 误差从 9.294 m 降到 0.231 m；这是输入数据纠错的收益。参考过渡尚未通过全量动力学资格，项目未完成实机闭环验收。完整统计、报告位置和失败边界见 [结果与能力限制](docs/results.md)。

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

Optional modes and their separate qualification limits: [native controller model](docs/design.md), [single-GPU multi-robot training](docs/design.md). Neither mode was used by the completed random 5000/2048 comparison.

## Preview

See [preview instructions](preview/README.md). Only fixed checkpoints may be selected. The server uses the same trained policy, history, observations, native action mapping and physics environment as training; reference ghost and actual policy are distinct. The configured release and checkpoint identity must be checked against `prepared/current.json` and the running service: [open preview](http://1.94.0.106:31769/). All 252 matched checkpoint evaluations are complete; [results and artifact locations](docs/results.md) include both improvements and remaining drift. Available releases are defined by the pinned catalog. See the [historical experiment summary](docs/results.md) for the separate 15k–30k run.

## Attribution

See [third-party notices](THIRD_PARTY_NOTICES.md). Code licensing does not replace robot asset or motion dataset terms. No Git commit is needed by the workflow.
