# 运行、部署与导出

## Runtime 与产物

项目根目录为 `/data/kk/blinkmorphmimic`，模块名保留 `morphmimic`。本地冻结依赖包含 Python 3.12、PyTorch 2.11/cu130、mjlab 1.6、MuJoCo/Warp；实际版本和补丁身份以运行镜像及收据为准。

```bash
docker build -t blinkmorphmimic:local .
```

直接 Python 调用使用 `PYTHONPATH=$PWD/src:$PWD`。外部 `/data/kk/data` 及 prepared/模型挂只读，输出和 cache 挂可写。新训练与恢复见 [训练](training-evaluation.md)。`reports/`、`runs/` 均不纳入 Git；部署凭据不得包含 `ops/private/` 认证信息。

## 预览身份与启动

预览地址为 [在线页面](http://1.94.0.106:31769/)。整理时 `prepared/current.json` 指向 `runs/inplace-mixed-muon-2048-7000to20000-20261118/preview-20000/releases.json`，其部署收据记录 20000 EMA 和 1280×720。实际在线状态需读取 `/api/status`，不能将历史收据当实时健康证明。

在 GPU runtime 中从根目录启动本地预览，先读取真实 current 配置：

```bash
DATA_ROOT=$(python -c 'import json,pathlib; print(pathlib.Path(json.load(open("prepared/current.json"))["manifest"]).parent)')
RELEASES=$(python -c 'import json; print(json.load(open("prepared/current.json"))["releases"])')
bash scripts/serve.sh --data-root "$DATA_ROOT" --releases "$RELEASES" \
  --device cuda:0 --compare-devices cuda:0,cuda:2,cuda:3 \
  --nvenc --encoder-gpu 1 --host 127.0.0.1 --port 31769
```

按实际设备分配 GPU，以上四卡安排不是所有主机的固定要求。本机 NVENC 验证需要全部 GPU 可见及 `compute,utility,graphics,video` 能力。NVENC 探测失败会拒绝启动。公网部署另需受控 host/public-host、31870–31889 UDP 发布与已有媒体代理；私密配置不进入文档或报告。

release 固定 checkpoint、训练清单、参考清单与人体文件身份。参考迁移后训练清单和播放清单分别绑定；不要修改 checkpoint 或历史训练清单去迎合新参考。起身窗口、中立位目标及翻转来源目录也需独立哈希校验。

## 使用与 API

选择 Motion/Human、单本体/同屏及动作后载入、播放；首次加载需要内核预热。暂停保留现场，重置回同一初始姿态。起身先选择躺/坐/半蹲入口再播放；翻转目录只筛选参考来源，策略是否成功以独立评价为准。

| API | 用途 |
|---|---|
| `GET /api/status` | release、checkpoint 身份、参考进度、真实物理与失败状态 |
| `GET /api/motions?robot=all&q=dance` | 动作查询 |
| `POST /api/control` | 选择、播放、暂停、重置与起身 |
| `POST /api/offer` | WebRTC 信令 |
| `GET /api/frame?format=png` | 原始 RGB 诊断；前端视频不轮询图片 |

同屏使用一个共享 actor 和独立本体物理世界；任一本体失败停止整组。策略实体与参考 ghost 分开。50 Hz 表示仿真控制时间，播放可慢于实时；分开报告物理 Hz、模拟倍速、渲染和解码 fps。所有浏览器共享物理会话，操作会互相影响；每观看端独立编码，多观看端共享编码已取消。

`--backend cpu --device cpu --compare-devices cpu` 可选原生 CPU 预览，拒绝 control graph。CPU 与 Warp 物理并不等价，曾有短窗口速度差超限，不能据此宣称 sim2sim 轨迹一致。GPU 的 `--control-graph` 也需目标配置单独验证；当前部署参数以其 launch 收据为准。

`--pico-data-root` 为人体记录提供持久目录，须配合 Human release。记录上传/回放不替代现场设备安装、比例和时间同步标定，也不构成在线实机遥操验收。

## 检查与恢复

```bash
curl -fsS http://127.0.0.1:31769/api/status
docker logs --tail 100 blinkmorph-preview
```

检查实际容器及镜像、发布清单 SHA、三本体 checkpoint、媒体和物理状态。需要重启时，在确认正式容器名称后执行 `docker restart blinkmorph-preview`。回滚使用对应部署收据中的旧容器与冻结镜像，避免套用过时 v12/v19 操作脚本；停/改名当前容器后恢复经确认的旧服务，再验证真实浏览器视频和模型身份。

## ONNX 导出契约

`python -m morphmimic.scripts.export` exports one numbered Q1, Bumi or G1
checkpoint to a new bundle. The default is its EMA policy. `--weights online`
selects online weights explicitly; a missing EMA is an error. `latest.pt`,
symlinks, mismatched filename/iteration and existing output directories are
refused. Checkpoints are trusted local PyTorch files, not an untrusted upload
format. The checkpoint hash is checked before and after loading.

```bash
PYTHONPATH=src python -m morphmimic.scripts.export \
  --checkpoint runs/shared-v1/q1/ckpt_1000.pt \
  --output exports/shared-v1-1000-ema/q1
```

Repeat for `bumi` and `g1`. This intentionally writes three graphs with the same
shared learned parameters and different fixed robot morphology, masks and
normalizers. Compare `shared_policy_sha256` across metadata files to establish
shared weights. This exporter does not claim that arbitrary checkpoint files
share their parameters. Each complete bundle is published only after CPU parity
passes and contains:

- `policy.onnx`: float32 graph, ONNX opset18, dynamic batch dimension.
- `metadata.json`: checkpoint identity, chosen weights, graph/shared parameter
  hashes, input contract, masks, morphology, normalizer statistics, IMU/anchor
  names, native mapping, PD gains, effort caps, action scales, default pose,
  compiled asset identity and measured parity errors.
- `training_robot_contract` records the saved collision/control contract when
  present. Older checkpoints return null; their asset-only hash does not prove
  the runtime kept all body contacts. Bumi before the iteration15000 correction
  used feet-only collisions and is not a whole-body-contact release.
- `SHA256SUMS`: graph and metadata hashes; check with `sha256sum -c SHA256SUMS`
  from the bundle directory.

The CPU export environment needs the project's existing optional ONNX tooling
(`onnx`, `onnxruntime`, `onnxscript`) plus its normal robot asset/mjlab dependencies.
All three robots' assets are needed to verify the shared morphology scaling. The frozen
training image need not contain ONNX Runtime: install it only into a separate
export environment. Keep assets read-only at `/data/kk/data`, or set
`BLINKMORPH_DATA_ROOT` to that root. Model MJB hashes include loaded meshes and
model parameters and are tied to the recorded MuJoCo version; they are not
hardware calibration certificates.

### Exact graph boundary

Both inputs are raw features **before normalization**. Morphology and running
normalizers are constants inside the graph. Batch sizes must match; inputs must
be finite float32 with these widths:

| Tensor | Shape | Meaning |
|---|---|---|
| `obs_history` | `[B, H*93]` | H observations, oldest first, last one current |
| `motion_goal` | `[B,670]` | 10 future frames of the position-free motion goal |
| `canonical_action` | `[B,29]` | Deterministic dimensionless action, absent slots exactly zero |

An observation frame is `[q_rel29, qd29, imu_gyro3, gravity3, previous_action29]`.
`q_rel` is canonical measured joint position minus the canonical default pose,
in radians; velocity is rad/s. Canonical position is
`sign[c]*(native_q[perm[c]]-offset[c])`; relative positions cancel this offset.
Gyro and projected unit gravity `[0,0,-1]` are in the recorded IMU body's frame.
The previous action is the previous graph output, before action scaling. Missing
canonical positions, velocities and previous actions should be zero. The actor
also masks those slots after normalization.

Each goal frame is `[reference_q29, reference_qd29, anchor_position3,
relative_anchor_rotation6d]`, using goal version2 and `goal_anchor=reference`.
Build it with the existing `model.rewards_torch.goal_features` convention: anchor
position in the first goal frame's reference pelvis heading frame (origin at
reference pelvis XY, absolute Z); orientation is relative to the robot anchor.
Goal frames are consecutive reference frames `f` through `f+9` at50Hz, held at
the final clip frame. Rotation6D is the first two rotation-matrix columns,
flattened row-major `[m00,m01,m10,m11,m20,m21]`. Joint values
are absolute canonical reference positions, not `q_rel`. Do not feed native
joint order, raw motion capture poses, absolute robot XYZ, Euler angles or an
arbitrary 6D rotation convention directly into this graph. The existing state
and reward feature helpers define the exact convention.

The position-free actor requires joint encoders, gyro, projected gravity and
robot anchor orientation (IMU orientation plus joint FK where needed). It does
not require global robot XYZ, but does not supply an orientation estimator,
motion retargeter, reference preview generator or raw human interface.

### Native position control is outside the graph

The output is a canonical action, **not native joint angles or motor torques**.
For each present canonical slot `c`, with native index `n=perm[c]`:

```python
native_target[n] = native_default[n] + sign[c] * canonical_action[c] * native_action_scale[n]
```

Skip `perm[c] == -1`; never index the last native joint with `-1`. Hold any
unmapped native joints at default. The JSON provides every value in this formula
and the exact native joint-name order. It preserves Bumi's negative elbow signs
and Q1's waist-roll/waist-yaw permutation. Bumi's native MJCF order differs from
its SDK/CSV order; map by name before writing a different transport. The graph
does not clip actions, targets, velocity or torque; runtime motor guards remain
the executor's responsibility.

Use 50Hz (20ms) policy updates and the recorded physics timestep/decimation for
simulation. PD gains/effort caps describe the nominal ideal position controller.
Q1's measured HAL delay queue, speed taper, slew, tilt guards and prediction reuse
are outside this graph, as are any training actuator-delay history. The graph
does not reproduce an execution wrapper. Retaining H observations and sampling
reference goals at the trained times is the caller's responsibility.

### Validation scope

Every export runs ONNX checker and compares ONNX Runtime CPU actions with the
same checkpoint's float32 PyTorch evaluation path at batch sizes1,4 and32.
Absolute maximum error must be at most1e-4, all outputs finite, and missing
actions exactly zero. Training BF16 autocast is deliberately disabled, matching
the existing evaluation loader. This is numerical export validation, not
tracking, fall-rate, sim2sim or hardware qualification.


Only the motion-only masked shared-checkpoint schema is supported. Joint Motion/Human checkpoints (`kp_interface=True`) are rejected by this exporter; the current 20000 release is not qualified for this export path. The separate upstream `deploy/export_onnx.py` has its own interface and is not interchangeable.

历史 `exports/shared-v2-contact-30000-ema/{q1,bumi,g1}` 为 motion-only 数值导出，CPU ORT 最大误差均 <1e-4；这不代表新联合 checkpoint 已导出，也不代表实际电机执行资格。
