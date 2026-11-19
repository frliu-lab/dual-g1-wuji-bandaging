# 双 G1 × Wuji 协作研究归档

**最新可展示的研究子技能（仿真）**：[19.2 秒双机托臂、抓卷轴与敷料拾取总览](media/bandage_v0/joint_dressing_pickup_sustained_one_second_demo.mp4) · [手部近景](media/bandage_v0/joint_dressing_pickup_sustained_one_second_demo_hand_closeup.mp4) · [逐帧验收](media/bandage_v0/joint_dressing_one_second_hold_pass_audit.json)。末段敷料被拇指和食指抬离台面并保持 1 秒，50 个控制帧全部通过托臂、双指接触、离台和碰撞门槛。视频由已验证的 15.7 秒路线与从相同状态重启的 3.5 秒控制续跑组成；卷轴仍在台面上。这是仿真局部子技能，尚未缠绕、固定带尾、实现四手完整协作或进行实机验证。更长的 1.5 秒持物尚未通过。

另有[17.7 秒托臂、夹敷料、握手柄并抬卷轴视频](media/bandage_v0/joint_handle_pickup_20mm_joint_pass_overview.mp4) · [运行与门槛](media/bandage_v0/joint_handle_pickup_20mm_joint_pass_manifest.json) · [50 Hz 关节轨迹](media/bandage_v0/joint_handle_pickup_20mm_joint_pass_qpos.npz)。该版本卷轴实际抬高约 18.79 mm，末段 A 左掌面承担约 96.2% 托臂冲量，但敷料仍在桌上。两种子技能尚未同时通过。较小幅度的[10 mm 卷轴版本](media/bandage_v0/joint_handle_pickup_first_joint_pass_overview.mp4)也已存档。

历史[30 mm 卷轴目标审计](media/bandage_v0/joint_handle_pickup_30mm_audit.json)通过数值门槛，但尚未录制/复核视频。早期[敷料拾取诊断](media/bandage_v0/joint_dressing_pickup_failure_audit.json)和[静态食指候选](media/bandage_v0/joint_dressing_index_tip_screen.json)记录了已被后续短时拾取方案改进的失败过程。后续需把卷轴悬空与敷料持物同时实现，再让带材绕臂、固定末端，并为 B 右手安排明确协作角色。

**新增局部进展：** [双指侧夹取敷料的检查点审计](media/bandage_v0/joint_dressing_side_pickup_checkpoint_audit.json)记录了 2.5 秒局部动态试验：A 右拇指与食指夹持敷料，敷料中心抬高 28.6 mm；A 左掌面末段承担 93.7% 托臂冲量，B 左卷轴漂移最大 1.64 mm，双机足部保持支撑且未发现深碰撞。末段 16 帧中 4 帧仍有极轻台面接触（峰值 0.0233 N），因此尚未通过严格连续离台门槛。试验从已有抓持状态重启，尚未整合进完整 17.7 秒路线。[侧向夹取静态候选](media/bandage_v0/joint_dressing_side_grasp_from_joint_screen_refined.json)及代码已存档。旧失败诊断是先前动作配置的实验记录。

可观看 [18.2 秒连续总览](media/bandage_v0/joint_dressing_pickup_checkpoint_continuous_demo.mp4)和[手部近景](media/bandage_v0/joint_dressing_pickup_checkpoint_continuous_demo_hand_closeup.mp4)。这两个视角来自同一 [50 Hz 状态轨迹](media/bandage_v0/joint_dressing_pickup_checkpoint_continuous_demo_qpos.npz)：前 15.7 秒为已验证的双机托臂、夹敷料和握卷路径，随后从关节位置完全一致的检查点重启控制器，继续 2.5 秒取出敷料。检查点速度也由保存状态恢复；请参见[演示清单](media/bandage_v0/joint_dressing_pickup_checkpoint_continuous_demo_manifest.json)。此版卷轴仍在台上，敷料尚有极轻台面接触，更没有绕臂、固定带尾或实机验证。

[持续夹持失败审计](media/bandage_v0/joint_dressing_sustained_hold_failure_audit.json)显示：试图在取出动作后额外保持 0.5 秒时，维持硬性向上受力约束会在约 2.64 秒失去可行解；撤销约束则双指脱离、敷料落回台面。提高软性夹持目标也未解决。因此上述视频只展示短暂的取出动作，不能解读为稳定持物或完整包扎。

**更新的可展示子技能：** [18.7 秒总览](media/bandage_v0/joint_dressing_pickup_sustained_half_second_demo.mp4) · [手部近景](media/bandage_v0/joint_dressing_pickup_sustained_half_second_demo_hand_closeup.mp4) · [逐帧验收](media/bandage_v0/joint_dressing_sustained_half_second_pass_audit.json) · [50 Hz 轨迹](media/bandage_v0/joint_dressing_pickup_sustained_half_second_demo_qpos.npz)。平滑减速并在末段调整 A 左托臂掌心后，敷料被双指抬离台面并保持 0.5 秒；25 个保持控制帧均有双指接触、无敷料台接触且托臂质量门槛通过。敷料抬高 31.6–32.8 mm，拇指/食指法向力分别在 0.647–0.696 N / 0.596–0.644 N；末段托臂掌面冲量占比 90.5%，卷轴漂移最大 2.0 mm。视频仍是从保存状态重启后的连续仿真回放；卷轴仍在台面上，尚无缠绕、固定带尾、B 右手协作或实机验证。上述失败审计保留为旧控制方案记录。

把相同抓持延长到 1.5 秒时，双指约在检查点续跑的 3.06 秒失去接触；见[长时保持失败审计](media/bandage_v0/joint_dressing_long_hold_failure_audit.json)。因此当前通过门槛明确限于 0.5 秒保持，不能推断持续搬运敷料或完成包扎。

[固定指尖目标方向的诊断](media/bandage_v0/joint_dressing_fixed_digit_frame_failure_audit.json)把双指脱离推迟到约 3.22–3.60 秒，但仍未同时通过长时夹持与托臂门槛。这个结果提示敷料旋转滑出是重要失效模式；暂不升级公开演示的验收结论。

**新的一秒保持验收：** 在取出末段冻结指尖目标方向，并分两阶段微调 A 左托臂掌心后，[审计](media/bandage_v0/joint_dressing_one_second_hold_pass_audit.json)确认 2.5–3.5 秒的 50 个保持帧全部有效。敷料离台 29.9–32.8 mm，拇指/食指法向力分别为 0.621–0.754 N / 0.582–0.697 N，无敷料台接触，末态托臂掌面占比 94.1%，卷轴漂移最大 2.13 mm。对应[19.2 秒视频](media/bandage_v0/joint_dressing_pickup_sustained_one_second_demo.mp4)、[近景](media/bandage_v0/joint_dressing_pickup_sustained_one_second_demo_hand_closeup.mp4)和[轨迹](media/bandage_v0/joint_dressing_pickup_sustained_one_second_demo_qpos.npz)已存档。继续到约 3.76 秒仍会失去双指接触，因此不能外推为长期持物或完整包扎。

继续研究更长的抓持时，[1.5 秒双指接触诊断](media/bandage_v0/joint_dressing_grip_support_tradeoff_audit.json)保持了 75/75 帧拇指与食指接触，但 A 左托臂仅 58/75 帧通过掌面承载门槛，末态掌面占比 80.8%。这一版本不通过联合门槛；公开可展示的最长联合验收仍是 1 秒。

[敷料向上受力不可行诊断](media/bandage_v0/joint_dressing_lift_force_infeasible_audit.json)进一步显示：在当前仿真接触与其他约束下，现有抓姿连很小的额外向上合力都无法由 WBC 满足。该结果不是硬件能力上限；需要重新设计 A 右手指腹接触与器具取用方式。

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

已新增[实际路由的解析式手指反馈](media/bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_roll0p05_relative0p6_routed_hold2p5_50hz_0.6s.json)：WBC 在仿真中持续运行，手部模式 83 个控制周期，静态接触连续 2.201 秒，末段接触比例 100%。这只通过**未离台接触**门槛；[延迟 5 mm 抬腕试验](media/bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_roll0p05_relative0p6_lift0p005_routed_liftstart1p2_hold2p5_50hz_0.6s.json)失去敷料。最新[21 组汇总](media/bandage_v0/pad_side_pinch_feedback_comparison.json)为未离台 1 组通过、带负载离台 0 组通过。策略尚不是训练好的 Wuji 手部模型，也不是完整四手包扎。

双手同场联合审计：[静态终态与预抓搜索](media/bandage_v0/joint_support_pinch_static_screen.json) · [零接触预抓路径](media/bandage_v0/joint_support_pregrasp_rrt_report.json) · [7.5 秒 WBC 基线](media/bandage_v0/joint_support_pinch_staged_wbc_report.json) · [11 组联合汇总](media/bandage_v0/joint_support_pinch_sweep_summary.json)。基线中 A 左手托臂在线门槛末段有效，但右拇指没有碰到敷料。提高右腕位置权重后出现双指接触，却使托臂门槛失败；联合门槛 0/11。B 机器人尚未参与操作，不能把此子任务称作四手协同包扎。

视觉核查：[静态终态](media/bandage_v0/joint_support_pinch_static_target_oblique.png)与[动态末态](media/bandage_v0/joint_support_pinch_dynamic_endpoint_oblique.png)并排可见右手实际到位偏差；静态图中的双指位置不能代替动态受力和持续接触。两张图只用于定位失败机理，不是任务演示。

[B 左手卷轴静态接触探测](media/bandage_v0/joint_spool_grasp_static_z0p745_screen.json)在原 B 根位置 x=0.70 m 未找到合格终态；[向内收紧 1.5 mm](media/bandage_v0/joint_spool_grasp_static_z0p745_closure0p0015_screen.json)出现 6 处深度接触。改为预置 B 根位置 x=0.55 m 后，[静态终态](media/bandage_v0/joint_spool_grasp_static_z0p745_root0p55_screen.json)达到双指相向接触、各约 1.33 N 且无深度碰撞；[三点进入路径](media/bandage_v0/joint_spool_grasp_rrt_report.json)通过静态筛查。[联合动态试验](media/bandage_v0/joint_support_pinch_staged_wbc_report_spool.json)在约 8.78/9.10 s 时因 QP 约束违背终止，B 双指未碰卷轴，A 右拇指未碰敷料，托臂在线门槛失效。B 的位置是预置的，尚无行走或动态抓卷轴证明；不能称为四手包扎。

定位到卷轴在动态运行时从静态设定的 z=0.720 m 落到约 0.702 m。[落稳卷轴静态终态](media/bandage_v0/joint_spool_grasp_static_z0p73_root0p55_spoolz0p702_screen.json)和[静态路径](media/bandage_v0/joint_spool_grasp_rrt_settled_report.json)均通过筛查。[完整 8.30 s 动态重放](media/bandage_v0/joint_support_pinch_staged_wbc_report_spool_settled.json)没有 QP 报错，双机足底接触 100%、无深度非足底碰撞；但 B 左食指仅峰值约 0.40 N、拇指未接触，A 右拇指也未碰敷料，A 左托臂末段门槛失效。[加大 B 指/腕跟踪权重](media/bandage_v0/joint_support_pinch_staged_wbc_report_spool_settled_bfinger10p0_bposition2000p0.json)也未恢复双指接触。两条动态试验都未达到包扎或四手协作门槛。

[B 左双指相对卷轴反馈](media/bandage_v0/joint_support_pinch_staged_wbc_report_spool_settled_bfeedback.json)使拇指短暂轻触，但没有持续相向抓持；[延长保持到 2.5 s](media/bandage_v0/joint_support_pinch_staged_wbc_report_hold2p5_spool_settled_bfeedback.json)仍无 B 左相向抓持，A 右敷料虽短暂相向接触约 0.62 s，托臂门槛已失效。这是解析式手指反馈研究试验，非训练好的 Wuji 小模型，也非可交付的包扎动作。

进一步按[动态末态卷轴姿态](media/bandage_v0/joint_spool_grasp_static_z0p73_closure0p004_root0p55_spoolz0p702_dynamicwarm_screen.json)重新求解 B 左手接触，避免把旧静态姿态直接移植后造成约 10–12 mm 的食指穿透。该静态候选虽无深度碰撞且双指相向，[8.30 s 闭环](media/bandage_v0/joint_support_pinch_staged_wbc_report_spool_settled_bwarm.json)中两指仅短暂接触，法线没有持续相向；[延长保持](media/bandage_v0/joint_support_pinch_staged_wbc_report_hold2p5_spool_settled_bwarm.json)也未改善。[收紧 8 mm 的静态候选](media/bandage_v0/joint_spool_grasp_static_z0p73_closure0p008_root0p55_spoolz0p702_dynamicwarm_screen.json)同样未带来动态抓持。

[托臂接触来源筛查](media/bandage_v0/joint_support_ring_clearance_screen.json)显示末段无名指近节承担约 25% 接触冲量：固定腕姿下 270 组指关节候选无一能在保留掌面接触时去掉该接触；729 组腕姿中 15 组静态几何合格。[较好的腕姿闭环](media/bandage_v0/joint_support_pinch_staged_wbc_report_spool_settled_bwarm_rewrist_roll.json)仅把末段手指冲量份额降到约 17%，仍超过暂定 10% 上限；[更紧的卷轴指姿组合](media/bandage_v0/joint_support_pinch_staged_wbc_report_spool_settled_bwarm_closure0p008_rewrist_roll.json)仍无持续相向抓持。静态接触筛查不能代替动态受力验收。

[放大腕姿偏移 1.3 倍](media/bandage_v0/joint_support_pinch_staged_wbc_report_spool_settled_bwarm_rewrist_roll_scale1p3.json)后手指冲量份额仍约 16.5%，B 左手相向抓卷轴时长仍为 0 s。此类末段角度微调未触及联合接触力分配的根因。

WBC 新增末段掌面法向力软目标，并修复了缓存 QP 中腕任务和关节加速度权重未随周期更新的问题。[11.5 N 掌面目标的 8.3 s 联合试验](media/bandage_v0/joint_support_pinch_staged_wbc_report_palmgoal11p5_spool_settled_bwarm_rewrist_roll.json)使 A 左托臂末段在线门槛通过（约 99.9% 掌面冲量、肘角 0.281 rad、峰值 18.53 N），但 A 右敷料与 B 左卷轴均未抓稳。[8 N](media/bandage_v0/joint_support_pinch_staged_wbc_report_palmgoal8p0_spool_settled_bwarm_rewrist_roll.json)、[10 N](media/bandage_v0/joint_support_pinch_staged_wbc_report_palmgoal10p0_spool_settled_bwarm_rewrist_roll.json)目标使前臂下垂，[12 N](media/bandage_v0/joint_support_pinch_staged_wbc_report_palmgoal12p0_spool_settled_bwarm_rewrist_roll.json)仍有约 12.7% 手指承力。

[分阶段提高 A 右腕权重并保持前臂肘角的联合闭环](media/bandage_v0/joint_support_pinch_staged_wbc_report_position1500p0_stageright_specialist_palmgoal11p5_elbowweight10p0_spool_settled_bwarm_rewrist_roll.json)完整运行 8.3 s：A 左托臂在线门槛通过，A 右敷料双指最长相向接触约 0.88 s，B 左双指都曾碰卷轴但相向接触仍为 0 s。解析式敷料手指专家实际运行 43 个控制周期。该试验仍未提起敷料、抓稳卷轴或绕带。按此末态重算的 B 左手[较高指尖目标](media/bandage_v0/joint_spool_grasp_static_z0p73_closure0p004_root0p55_spoolz0p702_dynamicwarm_jointwarm_screen.json)无法形成相向法线，[较低目标](media/bandage_v0/joint_spool_grasp_static_closure0p004_root0p55_spoolz0p702_dynamicwarm_jointwarm_screen.json)造成多处深碰撞与约 16.6 N 食指静态力。尚无可用的四手包扎策略。

B 左手多起点 IK 中，[种子 1](media/bandage_v0/joint_spool_grasp_static_z0p73_closure0p004_root0p55_spoolz0p702_dynamicwarm_jointwarm_seed1_screen.json)和[种子 2](media/bandage_v0/joint_spool_grasp_static_z0p73_closure0p004_root0p55_spoolz0p702_dynamicwarm_jointwarm_seed2_screen.json)在静态联合末态达到无深碰撞的相向接触；食指静态力约 4.25–4.35 N。[种子 1 完整闭环](media/bandage_v0/joint_support_pinch_staged_wbc_report_position1500p0_stageright_specialist_palmgoal11p5_elbowweight10p0_spool_settled_bwarm_jseed1_rewrist_roll.json)仍无 B 相向抓持；[增加 0.5 N 双指力目标](media/bandage_v0/joint_support_pinch_staged_wbc_report_position1500p0_stageright_specialist_palmgoal11p5_elbowweight10p0_spool_settled_bwarm_bforce0p5_rewrist_roll.json)也未改善。静态候选与力目标均不能替代闭环抓持证据。

逐接触点记录显示 B 左拇指指尖压在卷轴侧面，食指远节却压在上端盖；两个接触法线接近垂直，这解释了为什么同时有力却无法夹稳。[闭环报告](media/bandage_v0/joint_support_pinch_staged_wbc_report_position1500p0_stageright_specialist_palmgoal11p5_elbowweight10p0_spool_settled_bwarm_rewrist_roll.json)保留每次双指同时接触时的位置、身体部件和法线。[13 组端盖规避静态筛查](media/bandage_v0/joint_spool_cap_avoid_summary.json)均未同时达到无深度碰撞和相向接触。当前接近方向与手腕姿态需要重新规划，不能把双指轻触当作抓取。

把 B 机器人**预置**到 x=0.55、y=0.10 m 后，卷轴侧面目标 z=[0.720](media/bandage_v0/joint_spool_grasp_static_closure0p004_root0p55_rooty0p1_spoolz0p702_screen.json)、[0.725](media/bandage_v0/joint_spool_grasp_static_z0p725_closure0p004_root0p55_rooty0p1_spoolz0p702_screen.json)、[0.730 m](media/bandage_v0/joint_spool_grasp_static_z0p73_closure0p004_root0p55_rooty0p1_spoolz0p702_screen.json) 均有无深度碰撞的相向静态候选；[低](media/bandage_v0/joint_spool_grasp_rrt_side_report.json)、[中](media/bandage_v0/joint_spool_grasp_rrt_side_mid_report.json)、[高](media/bandage_v0/joint_spool_grasp_rrt_side_high_report.json)的静态进入路径均通过筛查。尚未规划或验证 B 行走到该站位。

[7 条侧向布局闭环汇总](media/bandage_v0/joint_side_layout_summary.json)：6 条通过暂定托臂和敷料接触子门槛，**0 条通过包含 B 卷轴低力持续相向抓持的联合门槛**。较低侧面目标在[10.1 s 重放](media/bandage_v0/joint_support_pinch_staged_wbc_report_position1500p0_stageright_specialist_hold2p5_palmgoal11p5_elbowweight10p0_spool_settled_side_rewrist_roll.json)中使 A 左托臂末段有效、A 右敷料双指相向接触最长 2.64 s，B 卷轴只有 0.04 s；[中等高度](media/bandage_v0/joint_support_pinch_staged_wbc_report_position1500p0_stageright_specialist_hold2p5_palmgoal11p5_elbowweight10p0_spool_settled_side_mid_rewrist_roll.json)仍只有 0.04 s 且 B 双指峰值约 7.12/5.71 N；[较高目标](media/bandage_v0/joint_support_pinch_staged_wbc_report_position1500p0_stageright_specialist_hold2p5_palmgoal11p5_elbowweight10p0_spool_settled_side_high_rewrist_roll.json)达到 0.52 s，但 B 双指峰值约 8.38/9.34 N，出现 13 个深度接触周期、卷轴位移约 22 mm，明确不合格。没有敷料离台和绕带证据。

静态筛选又发现一处质量漏洞：低/中目标的食指实际指尖高度都约 0.735 m，最大目标误差分别约 16.4/12.2 mm，导致动态仍从卷轴端盖接触。现在另列“严格静态”标记，要求指尖误差不超过 10 mm、单指静态力不超过 2 N 且无食指端盖接触；[低](media/bandage_v0/joint_spool_grasp_static_closure0p004_root0p55_rooty0p1_spoolz0p702_screen.json)、[中](media/bandage_v0/joint_spool_grasp_static_z0p725_closure0p004_root0p55_rooty0p1_spoolz0p702_screen.json)不通过，[高](media/bandage_v0/joint_spool_grasp_static_z0p73_closure0p004_root0p55_rooty0p1_spoolz0p702_screen.json)虽然静态通过，动态已显示高力与穿透。静态标记仍不是抓持资格。

[27 组 B 固定站位与高度网格](media/bandage_v0/joint_spool_layout_grid_summary.json)中仅一组通过严格静态筛选，而它的动态重放已因高力和穿透失败。继续在当前双指末态模板上调站位不具备可验证的成功候选；下一步需要整手接近姿态、卷轴表面接触点与法线的联合规划，以及训练并验证适配 Wuji 硬件的手部策略。
