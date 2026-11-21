# BlinkMorphMimic 项目指南

## 基础信息与文档入口

- 项目名 `blinkmorphmimic`，Python 包名 `morphmimic`，核心源自固定版本 MorphMimic；归属与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 正式支持 Q1 22、Bumi 21、G1 29 个原生关节。`robots/` 内保留的其他上游文件不等于已注册或已验证的本体。
- 环境要求以 [pyproject.toml](pyproject.toml)、[Dockerfile](Dockerfile) 和具体冻结镜像为准；Python 声明范围为 >=3.11,<3.14，本地已用 Python 3.12/CUDA runtime。
- 先看 [README.md](README.md) 的优化与实测，再按任务查 [设计](docs/design.md)、[机器人契约](docs/robot-contracts.md)、[数据](docs/data.md)、[训练评价](docs/training-evaluation.md)、[部署](docs/deployment.md)、[结果](docs/results.md)。避免新增重复设计或执行台账。
- 本地可变入口为 `prepared/current.json`，包含 manifest/SHA、人体/五点根目录、release 和训练运行。不能从日期、旧 README 或某次收据推断实时在线版本；通过实际配置及 `/api/status` 核对。

## 架构与代码归属

| 路径 | 职责 |
|---|---|
| `src/morphmimic/robots/` | 注册本体、资产、原生↔canonical 映射、形态特征与控制参数 |
| `src/morphmimic/data/` | 动作/人体准备、读取、重采样、配对与来源校验 |
| `src/morphmimic/env/` | mjlab/MuJoCo-Warp 物理、观测、动作、奖励、终止、reset 和延迟记录 |
| `src/morphmimic/model/` | 形态条件 WM、Motion/Human policy、目标特征、奖励及可选控制器状态 |
| `src/morphmimic/train/` | 采集/replay、模型想象与更新、共享梯度、采样、冻结身份和 checkpoint 恢复 |
| `src/morphmimic/eval/` | 固定策略加载、benchmark、checkpoint 批评及漂移统计 |
| `src/morphmimic/scripts/` | 包级训练、数据准备、评价与导出 CLI |
| `scripts/` | 优化训练、续训、数据迁移、身份审计、专项评价和预览启动入口 |
| `preview/` | 固定 release 目录、共享 actor、独立物理会话、CPU/Warp 后端和 WebRTC/NVENC |
| `deploy/` | 上游部署/导出接口；与本项目 motion-only ONNX 导出契约分开核对 |
| `tests/` | 正式测试唯一发现目录；GPU、资产和 checkpoint 相关用例可能有显式条件 |
| `docs/` | 正式设计、使用说明、结果摘要与许可 |

数据流为：原始动作/真实配对人体 → 校验后的 prepared manifest 与 NPZ/sidecar → 原生物理采集 → replay → WM 学习及 Motion/Human 想象更新 → 固定 checkpoint → 独立评价/预览/受支持的导出。

默认训练为三 rank distributed，每本体一张 GPU，参数共享、物理及校准 buffers 按本体保留。统一 29 槽状态及 154 维形态/mask（无 actor geometry 时 145 维）；Motion 与 Human 是不同输入路径，不能混用目标编码。可选 `single_gpu_sequential` 与 `native_delay_v1` 有单独的配置、恢复和资格边界。

## 关键入口与命令

从根目录设置 `PYTHONPATH=$PWD/src:$PWD`，需要原生 GPU/物理依赖的工作优先使用已有冻结 runtime。

| 入口 | 用途 |
|---|---|
| `python -m scripts.train_optimized --help` | 新随机初始化三本体 Muon/Adam 联合配方；先 dry-run，再冻结 source/config/image/data |
| `python -m morphmimic.scripts.train --help` | 通用训练及可选 executor/controller |
| `python -m scripts.train_inplace_20000 --help` | 校验固定父权重与迁移数据的原地起身续训 |
| `python -m scripts.evaluate_motion_suite --help` | selected/flip 独立物理评价 |
| `python -m scripts.evaluate_recovery --help` | 起身窗口和站姿判定 |
| `bash scripts/serve.sh --help` | 固定 release 的真实物理预览 |
| `python -m morphmimic.scripts.export --help` | 受支持 motion-only checkpoint 的 ONNX 导出 |

验证按改动范围选用对应 `tests/test_*.py`，例如映射改动运行 `PYTHONPATH=src:. python -m pytest -q tests/test_robots.py`。运行后如实报告通过与跳过条件；文档改动检查链接、路径和示例语法即可，不启动训练、占用 GPU 或重启服务来验证文档。

## 数据、控制和恢复约束

- 原始 `/data/kk/data`、外部源项目和机器人资产保持只读；新增数据输出到新目录，校验后再切换入口。保留来源、group、train/val、时间映射、manifest 及哈希。
- 五点人体监督必须来自真实配对人体；MotionDecode 仍为 motion-only。人体根局部轴不直接等同机器人朝前方向，位置和四元数使用一致的刚体变换。
- 缺失关节 `perm=-1` 不能索引最后一个 native 关节。输入/噪声/目标/损失/输出需保持 mask，Bumi SDK/XML 顺序和肘部符号按名称与约定处理；不降低校准或控制验证。
- 统一 50 Hz 控制，保留各本体 physics 子步、原有碰撞及故障终止。原生 overflow 为零不代表所有接触求解问题已覆盖。
- 固定编号 checkpoint 用于评价、导出和预览；`latest` 不作为这些用途的身份。恢复核对网络、优化器、EMA、normalizer、sampler、RNG 和数据/镜像/配置身份，replay 与物理会重新预热。
- 冻结身份覆盖 `docs/` 等源文件。历史恢复使用对应原源码/镜像，修改后新运行重新冻结；单 GPU 恢复需要完整 checkpoint 集合，不能仅拷一个本体文件或跨 executor 恢复。
- 标准站姿来源为 LAFAN1 `walk2_subject4` 第 11446 帧。严格原地起身需实际位置、朝向、高度、全身/手臂姿态达标并持续保持 3 秒；参考到终点、普通直立、短窗口和全动作成功分别记录。

## 文件与证据管理

- `docs/` 只保留正式说明与必要许可；临时脚本、日志、截图、原始评价和身份收据写入 `reports/`，训练产物写入 `runs/`。
- `reports/`、`runs/`、`prepared/`、`exports/`、cache 和 `ops/private/` 已被 Git 忽略，不强制加入版本管理，不把凭据、数据或 checkpoint 塞回文档目录。
- 优先复用已有 helper，修改前查调用者；同步受影响的契约、测试、入口与文档。不可把模拟器/导出/浏览器验收提升成实机资格，或把单 seed 组合收益说成单项优化的因果结论。
- Git 提交、推送和历史重写按用户明确要求执行；历史清理后不要通过旧分支或快照重新引入已移除的 docs 过程产物。
