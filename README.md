# 双 G1 × Wuji 协作研究归档

此仓库按实验批次存档双 G1、四只 Wuji 手协作研究。`media/` 保存当前可交流的代表性视频与对应说明；`VIDEO_CATALOG.json` 列出本地项目目前全部视频及 SHA-256，便于后续按实验批次补充。当前公开仓库只上传了五段代表性视频，其余历史视频尚未上传。在原始项目根目录运行 `python work/research_archive/build_archive_catalog.py` 可更新目录；公开仓库中的脚本副本位于 `code/work/research_archive/`。

当前可播放：[三动作合集](media/four_hand_dynamic_v2/three_cooperative_actions_conditional_sim.mp4) · [协同搬运](media/four_hand_dynamic_v2/long_box_obstacle/open_grasp_full_mass1p5_overview_conditional_sim.mp4) · [转桌](media/four_hand_dynamic_v2/table_doorway/turn_6deg_full_open_rest_v8_overview_conditional_sim.mp4) · [抬桌](media/four_hand_dynamic_v2/table_doorway/lift_12cm_full_open_rest_v12_overview_conditional_sim.mp4)。

新任务场景：[训练假人救护站预览](media/bandage_v0/scene_preview.png) · [两圈路线诊断图](media/bandage_v0/route_diagnostic.png) · [场景几何检查](media/bandage_v0/scene_check.json) · [卷轴扰动检查](media/bandage_v0/route_check.json) · [初始四手接触审计](media/bandage_v0/contact_evidence_initial.json)。路线是固定在假人手臂上的中心线，不是布料或已完成的包扎动作。

柔性带材材料试验：[双圈端态预览](media/bandage_v0/helix_flex_pilot_preview.png) · [端态检查](media/bandage_v0/helix_flex_pilot_check.json)。图中带材由夹具预先绕好并固定两端，机器人没有参与；它只验证材料接触模型的一个端态。

动态引导代理试验：[90° 引导慢放视频](media/bandage_v0/guided_flex_pilot_4x_slow.mp4) · [数值检查](media/bandage_v0/guided_flex_pilot_check.json) · [九组参数扫描](media/bandage_v0/guided_flex_sweep.json)。视频把 0.5 s 仿真以 4 倍慢动作播成 2 s；带材起始已预绕两圈，蓝点是运动学引导点，没有机器人手部动作。9/9 只表示固定前臂材料代理试验通过预设几何门槛。

四手腕 WBC：[单帧求解审计](media/bandage_v0/four_palm_wbc_probe.json) · [静止接地 50 Hz 站立 2 秒](media/bandage_v0/four_palm_wbc_stance_50hz_contact_2s.json) · [静止接地 200 Hz 站立 2 秒](media/bandage_v0/four_palm_wbc_stance_200hz_contact_2s.json) · [无控制落地失败对照](media/bandage_v0/four_palm_wbc_stance_200hz_drop_0.3s.json)。两机短时站立通过，但尚无包扎任务接触，也没有把柔性带材并入 WBC。

当前视频是自建场景中的 MuJoCo 条件仿真，不是 CORE4D、HumanoidMimicGen、SIMPLE 或 Xp-Teleop 的 benchmark 得分，也不是实机结果。主合集中的三个任务为协同搬运、转桌、抬桌。较新的低摩擦转桌视频是额外研究候选；2 ms 延迟修正候选目前只有轨迹和审计，尚未有对应视频。

新方向 `bandage_v0/` 目前只有训练假人救护站场景预览与初始几何审计；`code/work/bandage_hierarchy/` 是 WBC 与手部策略切换的控制接口原型和场景生成代码。包扎动作、布料物理与训练好的手部策略尚未完成，不得将静态预览说成任务演示。

归档规则：仓库存代码、说明、清单和代表性视频；其余历史视频按日期或实验批次逐步发布，并在 README 链接对应文件。不要将未核查授权的第三方数据、模型权重或人像素材混进归档。每次发布前核对清单哈希、来源标签、实验条件和通过／失败状态。

研究任务的当前入口是 `code/work/bandage_hierarchy/README.md`。场景预览仅表示场景结构已建立，并不表示包扎动作完成。

最新动态前臂对照：[独立场景模型](media/bandage_v0/dynamic_forearm_variant.xml) · [机制检查](media/bandage_v0/dynamic_forearm_variant_check.json) · [双机 WBC 反事实报告](media/bandage_v0/dynamic_forearm_support_comparison.json)。0.8 kg 被动前臂的无接触与小指侧接触两种条件都下垂到约 0.3507 rad；现有左手路径没有形成有效托举，仍判为失败。该结果是条件仿真，不代表真人受伤手臂或实机表现。

后续掌托候选：[静态接触搜索](media/bandage_v0/dynamic_palm_support_search_root_m0p3_pinky0p5.json) · [安全起点和路径检查](media/bandage_v0/dynamic_palm_support_rrt_report.json) · [闭环对照](media/bandage_v0/dynamic_palm_support_wbc_comparison.json) · [七组扫参汇总](media/bandage_v0/dynamic_palm_support_sweep_summary.json)。该候选在仿真中减小前臂下垂，但掌部承力不足、食指承担过多载荷且峰值力高；按暂定研究门槛 0/7 通过，**不可视作有效包扎或实机质量动作**。

更新的[16 组条件汇总](media/bandage_v0/dynamic_palm_support_sweep_summary.json)包含前臂跟随、力反馈、软接触和 0.5/0.8/1.2 kg 质量敏感性。0.8 kg 的[较弱跟随试验](media/bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5.json)末段由掌面承力，但约 37 N 的初碰峰值未解决；[0.5 kg](media/bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_mass0p5.json)与[1.2 kg](media/bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_mass1p2.json)均失去这一接触质量。**16 组仍为 0 组达标**。此前七组结果是历史快照，以更新汇总为准。

控制接口已增加托臂质量门槛：仅四手接触不足以切换到手部专家。门槛当前默认关闭，尚未接入实时机器人控制；不应把它表述为已完成大小模型控制闭环。

后续已将[在线质量门槛](code/work/bandage_hierarchy/support_quality.py)接入 MuJoCo 接触诊断，并用真实仿真时序回放大小模型路由器：0.8 kg 候选仍因初碰峰值超研究门槛而始终不放行。三组独立扶手布局也失败，详见[更新的 19 组汇总](media/bandage_v0/dynamic_palm_support_sweep_summary.json)。这是仿真与路由器接口验证，尚未接入实机传感器或训练好的手部策略。

[掌托正面诊断图](media/bandage_v0/dynamic_palm_candidate_front.png)和[俯视诊断图](media/bandage_v0/dynamic_palm_candidate_top.png)展示当前候选的接触姿态。前段加速进入路径的峰值力和食指受力进一步恶化；图像仅供查找失败机理，不是合格动作演示。

更近的 A 站位增加了[−0.25 m 动态对照](media/bandage_v0/dynamic_palm_support_wbc_comparison_rootm0p25.json)和[−0.20 m 路径失败检查](media/bandage_v0/dynamic_palm_support_rrt_root_m0p2_report.json)。前者峰值力和中指承力仍不合格，且进入幅度缩小 10% 即失去接触；后者只有静态终态、未找到路径。[最新 23 组汇总](media/bandage_v0/dynamic_palm_support_sweep_summary.json)仍为 0 组达标。

把假人肘角初态设为自然下垂后的 0.35 rad 再规划，也没有修好接触：[闭环报告](media/bandage_v0/dynamic_palm_support_wbc_comparison_rootm0p25_elbow0p35.json)显示末段仍由手指承力、峰值约 28 N。[更新的 24 组汇总](media/bandage_v0/dynamic_palm_support_sweep_summary.json)为 0 组通过；静态掌面接触不能替代动态整手验证。

最新 35 组托臂仿真汇总中，2 组通过暂定子技能门槛：[0.8 kg / 12 N 总力约束 / 2 N 手指约束](media/bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_wbcforce12p0_wbcfinger2p0.json)和[0.5 kg / 10 N / 2 N](media/bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_mass0p5_wbcforce10p0_wbcfinger2p0.json)。[1.2 kg / 14 N / 2 N](media/bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_mass1p2_wbcforce14p0_wbcfinger2p0.json)因峰值力 31.43 N 未通过。详见[更新汇总](media/bandage_v0/dynamic_palm_support_sweep_summary.json)。这仅是条件仿真的托臂子技能，不是完整包扎或实机验收。

新增[敷料相对位姿反馈试验](media/bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_roll0p05_relative0p6_hold2p0_50hz_0.6s.json)：2 秒保持的连续相向接触延长至 1.829 秒，末段比例 90%；延长到[2.5 秒](media/bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_roll0p05_relative0p6_hold2p5_50hz_0.6s.json)后接触失效。按[12 组更新汇总](media/bandage_v0/pad_side_pinch_feedback_comparison.json)仍为 0 组通过，敷料未离台，不可称为成功抓取。

[20 mm 离台诊断](media/bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_roll0p05_relative0p6_lift0p02_hold2p0_50hz_0.6s.json)中，敷料最大仅上升 3.73 mm，末态回到台面，接触失效且横移约 19 mm。更新汇总为 13 组、0 组通过；这是抓持失败的反证。
