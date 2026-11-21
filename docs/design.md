# 系统设计

## 范围与结构

项目沿用 MorphMimic 核心，支持 Q1（22 原生关节）、Bumi（21）、G1（29）。形态条件世界模型、Motion policy 和 critic 跨本体共享参数；Human 分支由真实配对人体监督训练。各本体保留原生物理、关节映射及校准 buffers。默认三 rank 分布式训练，每本体一张 GPU；优化配方每卡 2048 环境。

状态统一到 29 个 canonical 关节槽。154 维形态描述包含 29 维关节 mask；无 actor geometry 时为 145 维。缺失关节在输入、历史、噪声、目标、输出、奖励和世界模型损失中显式屏蔽，原生动作仅 scatter 到实际存在的关节。具体约束见 [机器人契约](robot-contracts.md)。

Motion 接口输入本体观测历史及参考动作目标；Human 接口使用真实人体骨架配对，常用五点为 pelvis、双脚踝、双手腕，H1。接口和权重配置须与 checkpoint 一致。MotionDecode 缺同源人体，只参与 Motion 路径；不使用机器人 FK 伪造人体监督。

## 学习与状态管理

世界模型从真实 replay 学习，策略通过模型想象优化。优化入口 `scripts/train_optimized.py` 使用 Muon 更新 WM/Motion/Human 内部矩阵，首末层和其他参数、两个 critic 使用 Adam；EMA 默认 0.995。世界模型就绪门槛、真实物理溢出检查和有限值检查保留。

优化配方包括 semi-implicit joint dynamics、reference reconstruction、contact/slip 奖励、显式五阶段学习图、collection graph、masked reset 和异步 checkpoint。启用选项不代表实验中必然触发：镜像增强受资产物理对称校验限制，随机 5000 实验实际镜像比率为零。实验质量和速度结论见 [结果](results.md)。

跨本体共享梯度等权平均；归一化与物理 buffers 按实际训练模式处理。固定 checkpoint 保存网络、优化器、EMA、normalizer、sampler、RNG 和覆盖计数。恢复后 replay 与物理世界重新初始化，不承诺逐位轨迹连续。源文件、镜像、配置和数据清单由冻结收据绑定。

续训的重要性采样结合来源分层探索与失败/几何误差 EMA；探索先均衡来源，再均衡来源内 clip/bin。访问次数表示实际使用，不表示独立轨迹数量、全库每帧覆盖或技能掌握。

## 可选原生控制器世界模型

`--controller-mode native_delay_v1` 显式建模模拟器内置 position controller 的延迟队列，默认 `legacy` 保留旧模型接口。它没有参与随机 5000 对照及当前本地 20000 运行，不应把默认训练已有执行器延迟与此可选条件模型混为一谈。

状态为 newest-first canonical 绝对弧度 target queue `[B,11,29]`（未用 lag 槽置零）与 int64 meta `[B,4]`（lag、push 次数、native step、phase offset）；真实记录包含前后队列、每子步 targets `[B,10,29]`、lags `[B,10]` 和全局 reset clock `[B]`。transition 必须接收显式未来 clock trace `[B,H]`，不能以零填缺失值。首次真实 reset write 初始化队列，首次 push 回填有效历史；观察记录本身不得改变队列或 RNG。实际 reset 还会令未重置世界额外写入 held target，故单世界队列不能独自封闭整个批量模拟器的状态。

想象从真实 replay 队列开始，bootstrap 连续观测到的全局 reset clocks；这只是给定外生时钟的条件预测。WM 使用 290 维有效目标特征并屏蔽旧 raw-action 输入块。策略输入与部署动作接口不变。只支持受校验的单一延迟位置执行器组、最大 lag 10、最多 10 子步等配置；不支持的设置拒绝执行。

支持新训练及同模式恢复，拒绝 legacy warm-start、隐式扩权重或重置优化器。恢复前严格核对 plant identity、控制映射和 buffers。2048×256 replay 每本体额外约 1.883 GiB，尚未证明大环境数容量及收益。小规模独立 CUDA fresh/resume 资格不能替代完整训练或硬件标定。

## 可选单 GPU 多本体训练

`--executor single_gpu_sequential` 在一张 CUDA 卡、一个进程中依次采集各本体，在每个 optimizer 边界等权合并梯度、只更新一次共享参数；物理、replay、采样器及校准 buffers 私有，normalizer 按样本数合并，EMA/target 每轮更新一次。

要求 `share=all`、pooled norms、Motion + Human return student 和 `latent_align_weight=0`。支持 Adam/Muon、compile none/default 及显式图；拒绝 reduce-overhead graph trees、跨 executor 恢复、部分共享、旧 teacher 初始化和 W&B。完整 schema2 集合通过 coordinator 和每本体文件 SHA 校验，不能只复制一个本体 checkpoint 恢复。

16 环境/本体小规模 fresh2→resume4 验证覆盖 legacy 与独立 native-delay 组合；未证明 2048×3 容量、最终学习质量或提速。默认仍为 distributed。恢复要求见 [训练与评价](training-evaluation.md)。

## 执行边界

预览运行固定策略和真实物理，reference ghost 与策略实体独立；三本体同屏每 tick 使用一个共享 actor、三行本体条件，身份或共享权重不一致即拒绝共享。任一参与者失败会停止整组。原生控制保持 50 Hz，墙钟播放速度需实测。

中立位过渡属于运动学参考；切换动作仍重建 simulation。连续实机切换、执行器动力学标定、PICO 现场标定与实机闭环均不能由预览或数据校验替代。
