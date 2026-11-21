# Physical preview

当前为单模型batch=3与NVENC WebRTC预览：[使用、实测与恢复](../docs/webrtc-preview.md)，[本轮完整证据](../docs/2026-11-12-batch-nvenc-results.md)。固定30000次EMA；同屏渲染为1280×720；物理和渲染分线程，性能需在目标环境实测。

在冻结GPU镜像中，从项目根目录启动（模型路径需使用真实固定checkpoint）：

```sh
scripts/serve.sh --data-root /workspace/blinkmorphmimic/prepared/full-v1 \
  --checkpoint q1=/models/q1/ckpt_30000.pt \
  --checkpoint bumi=/models/bumi/ckpt_30000.pt \
  --checkpoint g1=/models/g1/ckpt_30000.pt \
  --device cuda:0 --compare-devices cuda:0,cuda:2,cuda:3 \
  --nvenc --encoder-gpu 1 --host 0.0.0.0 --port 31769
```

本机容器需要全部GPU可见与compute,utility,graphics,video能力；公网部署还需要文档中的public-host及UDP映射。NVENC初始化失败会拒绝启动。单本体保留原load_policy，三本体同屏每tick一份共享actor，显式逐行本体条件；固定checkpoint SHA在启动和加载时核验。

API：GET /api/status、GET /api/motions?robot=all&q=dance；POST /api/control选择/播放/暂停/重置；POST /api/offer为WebRTC信令。GET /api/frame?format=png是源RGB诊断，默认为JPEG；前端视频不轮询图像。

所有浏览器共享一个物理会话。原始数据、模型只读；碰撞/PD/子步、关节映射和20ms控制dt保留。失败/片段结束会冻结，不能用预览代替完整评价或实机验证。

联合训练固定版本使用现有release字段，并额外提供 `training_manifest: {path, sha256}`：该文件的哈希必须与每个checkpoint的 `optimized_training.prepared_manifest_sha256` 一致。human版本的 `data_root` 仍须是完整原始人体配对目录；仅动作数据需使用单独的motion版本。固定checkpoint、训练清单和配对清单均在加载时校验。

`--control-graph` 可选启用分阶段CUDA图：先运行actor、核验动作，再运行无自动重置的物理图，并在每步检查原生溢出和终止；重置会重建物理图。仅支持不启用step/interval事件、记录器或观测NaN处理的环境，启动前须在目标GPU完成等价性验证。

### Native CPU preview

`--backend cpu` selects a native MuJoCo world and CPU actor; without `--device` it
uses `cpu` automatically. The default remains the Warp backend on `cuda:0`. CPU
preview supports one world per selected robot and the same shared three-robot actor.
It requires CPU actor/comparison devices and rejects `--control-graph`. The native
CPU backend uses the registered models, evaluation PD controls and 50 Hz timestep.
Reset joint positions obey the same native soft limits as the Warp reset command;
the original reference remains available for visualization. It is a preview path and does not change training or the existing Warp quality
evaluation. Native CPU warning and finite-state checks do not report Warp-only
overflow counters. CPU and Warp are independent numerical physics backends; do not
assume their trajectories are equivalent. A fixed 10-step CPU/Warp comparison failed
its preregistered 0.05 joint-velocity limit (Q1, second step: 0.271 at a contact).
This is not evidence that CPU preview itself cannot run, but it precludes a claim of
trajectory parity. Nine-case follow-up is diagnostic and does not replace a full
quality evaluation. Bumi short cases reached a maximum velocity difference of
1.435 rad/s, so the numerical difference is not limited to Q1. The online service
uses the separately validated Warp/NVENC path. Multi-viewer shared encoding has
been cancelled for the single-user deployment; each viewer uses an independent encoder.

### 起身演示（5000 checkpoint）

页面选择单本体或三本体同屏后，点击「站起来：载入倒地姿态」，待画面载入后点击「播放」。该入口固定使用 `joint-motion-5000`，从 LAFAN1 `fallAndGetUp2_subject2` 的第 650 帧（13 秒）初始化真实物理状态；此后由策略输出控制，跟踪剩余参考动作。「重置」会回到同一倒地姿态。它不表示任意倒地后的自主恢复能力，也不会额外训练或替换权重。

对应控制请求为 `POST /api/control`，JSON 为 `{"action":"standup","robot":"all","release":"joint-motion-5000"}`；`robot` 也可为 `q1`、`bumi`、`g1`。接口拒绝外部传入参考帧、动作路径或其他模型。状态中的 `initial_frame` 和 `reference_frame` 可核对起点与进度。单本体在顶层状态、同屏在 `participants` 中提供这两个字段。

私有同屏实测：G1 在第 104 控制步（2.08 秒）触发 `anchor_pos` 失败，三台随会话暂停；Q1、Bumi 此时未触发终止，这不能证明它们已完成起身。失败信息照常显示，不跳过、不自动重置。浏览器已验证倒地初始化、1280×720 H264/NVENC、播放、重置及重连；本功能未做实机验证。

公网 `http://1.94.0.106:31769/` 已验证同样结果：三本体分别运行于 CUDA 0/2/3，720p H264/NVENC、固定 checkpoint SHA、倒地重置和重连均通过，无页面、HTTP 或运行时错误。验收结束保留三台倒地初态，点击播放即可观察。证据位于 `runs/reports/preview-standup-20261114/`。

### 翻转目录与 7000 固定版本

已审核 release 可增加 `flip_sources: {path, sha256}`，绑定 `flip-preview.json` 与同一 `training_manifest`。目录按三个本体共有的 `source_id` 加入 BONES-SEED 原始翻转片段；Human 版本还校验真实人体源文件和重采样 sidecar。页面「翻转动作」只筛选这些片段，仍需点击「载入动作」；播放始终由固定策略驱动实际物理。参考翻转不代表策略已经学会翻转，成功率以实际旋转、离地和稳定落地评测为准。

7000 的起身入口继续使用审核后的自然垂手目标和原有初始姿态。其 `recovery_windows` 必须绑定当前训练清单；新版模型不会修改起身成功标准。
