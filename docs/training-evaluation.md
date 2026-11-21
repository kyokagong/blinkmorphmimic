# 训练与评价

## 训练入口

在项目 runtime 中运行，原始数据及 prepared 输入只读，`runs/`、`reports/` 和 cache 按需可写。构建与运行见 [部署](deployment.md)。

| 入口 | 用途 |
|---|---|
| `python -m scripts.train_optimized` | 新随机初始化优化配方，三卡各 2048 环境、5000 iterations，真实人体五点与 Motion 联训 |
| `python -m morphmimic.scripts.train` | 通用训练，默认 distributed，可显式选择 single_gpu_sequential |
| `python -m scripts.train_inplace_20000` | 已固定 7000 父权重在修正数据上续训至 20000，校验 parent/config/freeze |
| `bash scripts/train.sh` | 历史 Adam/30000 配方 |

从项目根目录设置 `PYTHONPATH=$PWD/src:$PWD`。运行前用 `--help` 确认参数，选择空闲 GPU 和新 run 名；不要同时向同一目录启动多个训练器。根 README 给出新随机训练的冻结及启动命令。

新优化配方的 WM 为 1024×3、每轮 5 次 batch1024 更新；policy 为 2048/1024/512/256，Motion/Human 使用同一实际采集系统。Muon/Adam、图优化和控制器模式以运行 config 为准。自适应 WM 就绪门槛和物理 overflow 停机不能绕过。`--dry-run` 只打印配置，不训练。

## 冻结与恢复

冻结收据绑定 source、config、不可变 image ID、manifest；启动和 resume 校验同一身份。新训练不传 resume 或初始化权重。改变数据/配方的续训需要显式 parent 和迁移凭据，不能冒充完全相同实验的恢复。冻结身份包含 `docs/`，因此这次整理后的工作区不能冒充旧冻结源码；历史 resume 使用原运行保存的源码快照/镜像，新运行重新冻结。

checkpoint 恢复网络、五个优化器、schedule、EMA、normalizer、sampler、RNG 及覆盖信息。replay、物理世界和 episode 重新初始化并预热，非逐位仿真续接。恢复保留原冻结配置与运行目录；评测、导出、预览使用固定 `ckpt_N.pt`，不使用 `latest.pt`。禁止拿 inference-only policy 恢复训练。

单 GPU 模式须提供完整 schema2 集合：每本体 full/policy/coverage、coordinator 与 `commits/iteration_N.json` 的身份一致。以原参数、run-name/root 加 `--resume`；可给集合目录或其中一个固定 full checkpoint，实际恢复全套。跨 executor 和旧 schema 自动迁移不支持；模式边界见 [设计](design.md)。

## 评价入口与口径

| 入口 | 范围 |
|---|---|
| `scripts/evaluate_optimized.py`、`watch_evaluations.py`、`plot_optimized.py` | 固定 checkpoint 的优化对照、汇总与曲线 |
| `scripts/evaluate_motion_suite.py` | `selected` 动作或 `flips` 专项，显式 checkpoint/manifest/robot/output |
| `scripts/evaluate_recovery.py` | 起身窗口，显式 windows、subset、站姿与数据迁移凭据 |
| `morphmimic.scripts.eval` | 通用完整动作评价 |

评价保存 checkpoint、清单、subset、资产、脚本身份、种子、精度、环境数和物理诊断；报告写入 `reports/` 或对应 `runs/<run>/reports/`。记录实际执行范围，不能用短窗口成绩代替完整长序列。

- `norm_return` 以完整片段长度归一化，提前失败不缩短分母。`source_norm_return` 仅在有 phases 时累计主体，主体前失败记零；没有 phases 时等于全片段。Q1 历史 entry/return/stand 不应混作主体。
- held-out 分区与训练探针分开；`all.yaml` 不等于全部未见动作。训练回报不能代替独立评价，也不能跨不同 reward 配方直接比较。
- 普通站稳、严格原地终态和连续保持 3 s 分别记录。严格起身要求位置误差 <10 cm、朝向 <10°、高度 <8 cm、全身及手臂关节 RMSE <0.15 rad，并按报告定义持续保持。
- 空翻需实际根旋转、腾空及稳定落地等独立物理证据；参考做了翻转、关节误差小或 episode 完成不等于空翻成功。
- 数据加载或采样/优化计数不等于逐帧覆盖、独立样本数量或动作掌握。零 native overflow 不排除未覆盖的 EPA/contact 问题。

probe 的时间账本绑定 training target，旧无 target 账本或有产物但缺账本的目录拒绝重用；保留旧产物并恢复原账本。默认 checkpoint 网格为 0、1000…5000，历史运行需显式给出迭代列表。

当前数据及参考身份见 [数据](data.md)，已观测结果与能力限制见 [结果](results.md)。上述入口说明不代表所有参数组合均已完成容量、学习质量或实机资格验证。
