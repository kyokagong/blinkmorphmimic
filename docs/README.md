# BlinkMorphMimic 文档

Q1、Bumi、G1 共享形态条件世界模型及动作跟踪策略，支持机器人动作参考和真实配对人体五点输入。

| 文档 | 内容 |
|---|---|
| [系统设计](design.md) | 最终架构、接口、默认路径与可选模式 |
| [机器人契约](robot-contracts.md) | 原生关节、控制、校准及物理诊断边界 |
| [数据](data.md) | 数据来源、当前清单、中立位与人体坐标约束 |
| [训练与评价](training-evaluation.md) | 训练入口、冻结恢复、评价与成功标准 |
| [部署](deployment.md) | 预览、运行恢复、API 与 ONNX 契约 |
| [结果与限制](results.md) | 重要实验结论与尚未达到的能力 |

本地数据入口为 `prepared/current.json`。整理时它指向修正后的原地起身数据及 20000 checkpoint 发布清单；这是本地配置与历史收据记录，不是实时服务状态保证。部署前核对实际 `/api/status`、发布清单和 checkpoint 身份。

`docs/` 只存正式文档和第三方许可。评价、诊断、截图及身份收据放在 `reports/`，训练日志与 checkpoint 放在 `runs/`；两者均已由根目录 `.gitignore` 排除，当前无 Git 跟踪文件。清理后保留的旧文档证据位于 `reports/docs-cleanup-20261121/`，其中 `inventory.json` 记录迁移、合并与删除前的路径及 SHA256。本地报告路径在纯源码检出中可能不存在。

许可证保留在 [licenses/](licenses/)，权利边界见 [第三方声明](../THIRD_PARTY_NOTICES.md)。数据可加载、参考正确、预览可播放和策略动作成功分别评价；仿真结果不构成实机资格。
