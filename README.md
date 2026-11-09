# 双 G1 × Wuji 协作研究归档

此仓库按实验批次存档双 G1、四只 Wuji 手协作研究。`media/` 保存当前可交流的代表性视频与对应说明；`VIDEO_CATALOG.json` 列出项目目前全部视频及 SHA-256，便于后续按实验批次补充。生成或更新目录：`python work/research_archive/build_archive_catalog.py`。

当前视频是自建场景中的 MuJoCo 条件仿真，不是 CORE4D、HumanoidMimicGen、SIMPLE 或 Xp-Teleop 的 benchmark 得分，也不是实机结果。主合集中的三个任务为协同搬运、转桌、抬桌。较新的低摩擦转桌视频是额外研究候选；2 ms 延迟修正候选目前只有轨迹和审计，尚未有对应视频。

新方向 `bandage_v0/` 目前只有训练假人救护站场景预览与初始几何审计；`code/work/bandage_hierarchy/` 是 WBC 与手部策略切换的控制接口原型和场景生成代码。包扎动作、布料物理与训练好的手部策略尚未完成，不得将静态预览说成任务演示。

归档规则：仓库存代码、说明、清单和代表性视频；其余历史视频按日期或实验批次逐步发布，并在 README 链接对应文件。不要将未核查授权的第三方数据、模型权重或人像素材混进归档。每次发布前核对清单哈希、来源标签、实验条件和通过／失败状态。

研究任务的当前入口是 `code/work/bandage_hierarchy/README.md`。场景预览仅表示场景结构已建立，并不表示包扎动作完成。

