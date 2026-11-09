# 双 G1 × Wuji 协作研究归档

此仓库按实验批次存档双 G1、四只 Wuji 手协作研究。`media/` 保存当前可交流的代表性视频与对应说明；`VIDEO_CATALOG.json` 列出本地项目目前全部视频及 SHA-256，便于后续按实验批次补充。当前公开仓库只上传了五段代表性视频，其余历史视频尚未上传。在原始项目根目录运行 `python work/research_archive/build_archive_catalog.py` 可更新目录；公开仓库中的脚本副本位于 `code/work/research_archive/`。

当前可播放：[三动作合集](media/four_hand_dynamic_v2/three_cooperative_actions_conditional_sim.mp4) · [协同搬运](media/four_hand_dynamic_v2/long_box_obstacle/open_grasp_full_mass1p5_overview_conditional_sim.mp4) · [转桌](media/four_hand_dynamic_v2/table_doorway/turn_6deg_full_open_rest_v8_overview_conditional_sim.mp4) · [抬桌](media/four_hand_dynamic_v2/table_doorway/lift_12cm_full_open_rest_v12_overview_conditional_sim.mp4)。

新任务场景：[训练假人救护站预览](media/bandage_v0/scene_preview.png) · [两圈路线诊断图](media/bandage_v0/route_diagnostic.png) · [场景几何检查](media/bandage_v0/scene_check.json) · [卷轴扰动检查](media/bandage_v0/route_check.json) · [初始四手接触审计](media/bandage_v0/contact_evidence_initial.json)。路线是固定在假人手臂上的中心线，不是布料或已完成的包扎动作。

柔性带材材料试验：[双圈端态预览](media/bandage_v0/helix_flex_pilot_preview.png) · [端态检查](media/bandage_v0/helix_flex_pilot_check.json)。图中带材由夹具预先绕好并固定两端，机器人没有参与；它只验证材料接触模型的一个端态。

动态引导代理试验：[90° 引导慢放视频](media/bandage_v0/guided_flex_pilot_4x_slow.mp4) · [数值检查](media/bandage_v0/guided_flex_pilot_check.json) · [九组参数扫描](media/bandage_v0/guided_flex_sweep.json)。视频把 0.5 s 仿真以 4 倍慢动作播成 2 s；带材起始已预绕两圈，蓝点是运动学引导点，没有机器人手部动作。9/9 只表示固定前臂材料代理试验通过预设几何门槛。

当前视频是自建场景中的 MuJoCo 条件仿真，不是 CORE4D、HumanoidMimicGen、SIMPLE 或 Xp-Teleop 的 benchmark 得分，也不是实机结果。主合集中的三个任务为协同搬运、转桌、抬桌。较新的低摩擦转桌视频是额外研究候选；2 ms 延迟修正候选目前只有轨迹和审计，尚未有对应视频。

新方向 `bandage_v0/` 目前只有训练假人救护站场景预览与初始几何审计；`code/work/bandage_hierarchy/` 是 WBC 与手部策略切换的控制接口原型和场景生成代码。包扎动作、布料物理与训练好的手部策略尚未完成，不得将静态预览说成任务演示。

归档规则：仓库存代码、说明、清单和代表性视频；其余历史视频按日期或实验批次逐步发布，并在 README 链接对应文件。不要将未核查授权的第三方数据、模型权重或人像素材混进归档。每次发布前核对清单哈希、来源标签、实验条件和通过／失败状态。

研究任务的当前入口是 `code/work/bandage_hierarchy/README.md`。场景预览仅表示场景结构已建立，并不表示包扎动作完成。
