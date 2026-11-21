# 数据与参考约束

## 当前入口与来源

以项目根目录 `prepared/current.json` 为本地入口，读取 manifest、SHA256、human/five-point 根目录及 release catalog；不要按目录日期猜测当前数据。整理时清单为 `prepared/inplace-recovery-neutral-20261117-v2/manifest.json`，SHA256 为 `8e77139fa7ddfef4cb2bb7fb9977d1f0fdcad30c94778ecc16c919969253547e`。

| 来源 | 使用与边界 |
|---|---|
| BONES-SEED | 原生机器人参考及真实 SOMA 配对人体；原始资产位于 `/data/kk/data` |
| LAFAN1 | 原始 BVH 与各本体重定向，包含真实起身来源 |
| MotionDecode | 原生机器人参考；缺少同源人体，保持 motion-only |
| 起身/翻滚增强 | 保留来源及时间映射，区分真实动作和合成过渡 |

原始数据、外部项目及资产只读，生成文件写入本项目 `prepared/`。初始 full-v1 为每本体 11006 条 SEED + 40 条 LAFAN1；每本体 9972 train、1074 val（含 2 条 LAFAN1）。同源家族、performer、镜像和跨本体 retarget 使用同一 group，保持分区隔离。旧全量 LAFAN1 40 条含 38 条训练内数据，不能统称未见动作。

当前混合数据共 34621 条机器人记录，其中修正起身 88 条；人体/五点配对 33244 条，1377 条 MotionDecode 为 motion-only。33 条不合格历史起身衍生记录退出训练，原始完整动作保留。当前清单与旧训练清单分别保存，不能把重写的数据当作旧运行的原始输入。

## 格式与转换

原生 NPZ 保存 native joint_names、14 个 tracked bodies、wxyz 四元数及位置/速度。输入 xyzw 显式转换，按真实源时间重采样至 50 Hz。MotionDecode 本项目使用已确认的 120 Hz 源约定。Bumi CSV/SDK 列必须按名称重排到 XML 原生顺序，不能只核对名称集合；源 elbow 符号和 root/FK/速度/足底/碰撞校验必须保留。

普通 MotionDecode 导入与 Bumi 重建复用 `data/motiondecode.py` 的转换/校验入口。旧 conversion receipt 被拒绝；校验失败不覆盖原目标、不发布失败文件或最终清单。历史修复中 464 条 Bumi 全部完成重建；这不表示所有原始来源都可用。

YAML `root` 相对于 YAML 文件解析。历史 manifest 的 output 可能记录容器路径；实际读取需以当前 NPZ 路径与 robot/name 匹配元数据。主体评分依赖 manifest，丢失它会退回全片段口径。人体五点来自真实人体，不使用机器人 FK 替代。

## 统一中立位与原地起身

标准为 LAFAN1 `walk2_subject4` 50 Hz 第 11446 帧（228.92 s），三本体各使用对应原生姿态：直立、双脚稳定、双手自然垂在大腿两侧。原始审批/目标及哈希随数据 provenance 保存。仅对齐世界朝向、XY 与地面高度，不改标准关节角。

普通动作采用中立保持至少 1 s → 平滑进入 → 原动作 → 平滑返回 → 中立保持至少 1 s；返回保持动作结束位置/朝向。原处理中段整数采样与速度保留，只延展获准的首尾段；不裁剪原始动作关节以掩盖超限，合成过渡不引入新极值。速度从最终保存精度重算。

原地起身采用单独约束：躺/坐保留首次持续近直立前的真实前缀，按站立锚点做固定刚体变换；半蹲从真实初姿平滑回中立位。终点与普通初始化的原生全身位置、四元数和关节一致，速度零，至少保持 3 s。preview 起身入口从标注初姿开始，并冻结最终参考，不停在旧中间帧。

人体根 X 轴沿左右髋，不是机器人朝前方向。人体根位置、四元数、平移元数据和五点导出使用同一个完整固定刚体旋转，防止约 90° 朝向偏差。人体骨长与源时间映射不变。

`time_s` 是均匀 50 Hz 播放时钟；`source_frame_index` 指上版处理帧。合成段 `source_time_s` 为 NaN，缺可信源时钟的衍生记录继续标未知。配对人体与机器人共享时间映射。

## 准备与复用

已有通过校验的数据无需重建。按用途使用现有入口：`morphmimic.scripts.prepare`、`prepare_human`、`prepare_motiondecode`，以及 `scripts/prepare_unified_neutral.py`、`prepare_inplace_recovery.py`。先查看对应 `--help`，输出到新目录，保留 manifest、source/group、分区和 provenance；校验完成后才更新本地入口。

旧 prepared 载荷曾被退役，历史训练恢复须核对文件仍在及数据迁移凭据，不能仅依靠旧路径存在。运动学通过不代表全量接触、碰撞、力矩或动态平衡合格。数据与资产权利见 [第三方声明](../THIRD_PARTY_NOTICES.md) 和 [licenses](licenses/)；代码许可不授予数据分发权。
