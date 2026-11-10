"""Build a read-only catalog and a small review bundle for the project's GitHub archive."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'wuji_multiagent'
DEST = ROOT / 'research_archive'
SELECTED = [
    'four_hand_dynamic_v2/three_cooperative_actions_conditional_sim.mp4',
    'four_hand_dynamic_v2/THREE_ACTIONS_README.md',
    'four_hand_dynamic_v2/THREE_ACTIONS_MANIFEST.json',
    'four_hand_dynamic_v2/long_box_obstacle/open_grasp_full_mass1p5_overview_conditional_sim.mp4',
    'four_hand_dynamic_v2/table_doorway/turn_6deg_full_open_rest_v8_overview_conditional_sim.mp4',
    'four_hand_dynamic_v2/table_doorway/lift_12cm_full_open_rest_v12_overview_conditional_sim.mp4',
    'four_hand_dynamic_v2/table_doorway/turn_6deg_full_open_rest_mu07_mass12_open3_footcap50_pregrip110_overview_conditional_sim.mp4',
    'four_hand_dynamic_v2/table_doorway/TURN_MU07_MASS12_FOOTCAP35_DELAY2MS_CANDIDATE.md',
    'four_hand_dynamic_v2/table_doorway/TURN_MU07_MASS12_FOOTCAP35_DELAY2MS_CANDIDATE_MANIFEST.json',
    'four_hand_dynamic_v2/REAL_HARDWARE_READINESS.md',
    'bandage_v0/scene_preview.png',
    'bandage_v0/scene_check.json',
    'bandage_v0/route_diagnostic.png',
    'bandage_v0/route_check.json',
    'bandage_v0/contact_evidence_initial.json',
    'bandage_v0/flex_pilot.xml',
    'bandage_v0/flex_pilot_check.json',
    'bandage_v0/helix_flex_pilot.xml',
    'bandage_v0/helix_flex_pilot_check.json',
    'bandage_v0/helix_flex_pilot_preview.png',
    'bandage_v0/guided_flex_pilot.xml',
    'bandage_v0/guided_flex_pilot_check.json',
    'bandage_v0/guided_flex_pilot_preview.png',
    'bandage_v0/guided_flex_pilot_4x_slow.mp4',
    'bandage_v0/guided_flex_sweep.json',
    'bandage_v0/guided_flex_sweep_baseline.json',
    'bandage_v0/four_palm_wbc_probe.json',
    'bandage_v0/four_palm_wbc_stance_50hz_drop_0.3s.json',
    'bandage_v0/four_palm_wbc_stance_200hz_drop_0.3s.json',
    'bandage_v0/four_palm_wbc_stance_50hz_contact_2s.json',
    'bandage_v0/four_palm_wbc_stance_200hz_contact_2s.json',
    'bandage_v0/reach_ik_probe.json',
    'bandage_v0/reach_ik_seed_preview.png',
    'bandage_v0/four_arm_reach_50hz_0.5s.json',
    'bandage_v0/four_arm_reach_50hz_0.5s.mp4',
    'bandage_v0/pad_touch_scan.json',
    'bandage_v0/pad_thumb_scan.json',
    'bandage_v0/pad_pinch_scan.json',
    'bandage_v0/pad_side_pinch_scan.json',
    'bandage_v0/pad_side_pinch_rrt_report.json',
    'bandage_v0/pad_side_pinch_attempt_2_50hz_0.6s.json',
    'bandage_v0/pad_side_pinch_attempt_2_50hz_0.6s.mp4',
    'bandage_v0/pad_side_pinch_feedback_comparison.json',
    'bandage_v0/pad_side_pinch_balance_2_gain0p0015_50hz_0.6s.json',
    'bandage_v0/pad_side_pinch_balance_2_gain0p0015_rot100p0_50hz_0.6s.json',
    'bandage_v0/pad_side_pinch_balance_2_gain0p0015_rot300p0_50hz_0.6s.json',
    'bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_50hz_0.6s.json',
    'bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_rollm0p05_50hz_0.6s.json',
    'bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_roll0p05_50hz_0.6s.json',
    'bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_roll0p1_50hz_0.6s.json',
    'bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_roll0p05_late0p025_50hz_0.6s.json',
    'bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_roll0p05_trace_50hz_0.6s.json',
    'bandage_v0/pad_side_pinch_balance_1_gain0p0015_rot100p0_roll0p05_50hz_0.6s_hand.mp4',
    'bandage_v0/forearm_support_root_m0p58_curl0p4_scan.json',
    'bandage_v0/forearm_support_root_m0p58_curl0p4_wrist_scan.json',
    'bandage_v0/forearm_support_root_m0p58_curl0p4_forwardarm_scan.json',
    'bandage_v0/forearm_support_root_m0p58_curl0p4_wrist_forwardarm_scan.json',
    'bandage_v0/forearm_support_rrt_report.json',
    'bandage_v0/forearm_support_wbc_report.json',
    'bandage_v0/forearm_support_wbc_trace.json',
    'bandage_v0/dynamic_forearm_variant.xml',
    'bandage_v0/dynamic_forearm_variant_check.json',
    'bandage_v0/dynamic_forearm_support_comparison.json',
    'bandage_v0/dynamic_palm_support_search_root_m0p3_pinky0p5.json',
    'bandage_v0/dynamic_palm_support_rrt_report.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison.json',
    'bandage_v0/dynamic_palm_support_sweep_summary.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_mass0p5.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_mass1p2.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_solref0p04.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_armrest.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_armrest0p775.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_armrestproximal0p8.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_easingfrontloaded.json',
    'bandage_v0/dynamic_palm_candidate_front.png',
    'bandage_v0/dynamic_palm_candidate_top.png',
    'bandage_v0/dynamic_palm_support_search_root_m0p2_pinky0p5.json',
    'bandage_v0/dynamic_palm_support_rrt_root_m0p2_report.json',
    'bandage_v0/dynamic_palm_support_search_root_m0p25_pinky0p5.json',
    'bandage_v0/dynamic_palm_support_rrt_root_m0p25_report.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_rootm0p25.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_rootm0p25.json',
    'bandage_v0/dynamic_palm_support_wbc_comparison_reach0p9_rootm0p25.json',
    'bandage_v0/pad_pinch_attempt_50hz_0.5s.json',
    'bandage_v0/pad_pinch_attempt_50hz_0.5s.mp4',
]
SOURCE_CODE = [
    'work/research_archive/build_archive_catalog.py',
    'work/bandage_hierarchy/README.md',
    'work/bandage_hierarchy/controller_router.py',
    'work/bandage_hierarchy/check_router.py',
    'work/bandage_hierarchy/build_scene.py',
    'work/bandage_hierarchy/render_scene_preview.py',
    'work/bandage_hierarchy/check_route.py',
    'work/bandage_hierarchy/scene_evidence.py',
    'work/bandage_hierarchy/check_scene_evidence.py',
    'work/bandage_hierarchy/flex_pilot.py',
    'work/bandage_hierarchy/helix_flex_pilot.py',
    'work/bandage_hierarchy/guided_flex_pilot.py',
    'work/bandage_hierarchy/render_guided_flex_video.py',
    'work/bandage_hierarchy/sweep_guided_flex.py',
    'work/bandage_hierarchy/check_wbc_adapter.py',
    'work/bandage_hierarchy/probe_four_palm_wbc.py',
    'work/bandage_hierarchy/rollout_four_palm_wbc.py',
    'work/bandage_hierarchy/reach_ik_probe.py',
    'work/bandage_hierarchy/render_reach_seed.py',
    'work/bandage_hierarchy/search_guide_waypoint.py',
    'work/bandage_hierarchy/rollout_reach_wbc.py',
    'work/bandage_hierarchy/probe_pad_touch.py',
    'work/bandage_hierarchy/probe_pad_thumb.py',
    'work/bandage_hierarchy/probe_pad_pinch.py',
    'work/bandage_hierarchy/probe_pad_side_pinch.py',
    'work/bandage_hierarchy/plan_side_pinch_path.py',
    'work/bandage_hierarchy/summarize_side_pinch_feedback.py',
    'work/bandage_hierarchy/probe_forearm_support.py',
    'work/bandage_hierarchy/plan_forearm_support_path.py',
    'work/bandage_hierarchy/rollout_forearm_support_wbc.py',
    'work/bandage_hierarchy/build_dynamic_forearm_variant.py',
    'work/bandage_hierarchy/compare_dynamic_forearm_support.py',
    'work/bandage_hierarchy/search_dynamic_palm_support.py',
    'work/bandage_hierarchy/plan_dynamic_palm_support.py',
    'work/bandage_hierarchy/summarize_dynamic_palm_support.py',
    'work/bandage_hierarchy/support_quality.py',
    'work/bandage_hierarchy/render_dynamic_palm_candidate.py',
    'work/wuji_multiagent/joint_wbc.py',
    'work/bandage_hierarchy/task_contract.json',
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    DEST.mkdir(exist_ok=True)
    all_videos = sorted(OUT.rglob('*.mp4'))
    catalog = {
        'scope': 'Project-generated videos only; catalog does not imply benchmark or hardware qualification',
        'video_count': len(all_videos),
        'total_bytes': sum(p.stat().st_size for p in all_videos),
        'videos': {p.relative_to(OUT).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in all_videos},
    }
    (DEST / 'VIDEO_CATALOG.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    selected = {}
    for name in SELECTED:
        source = OUT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = DEST / 'media' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != digest(source):
            shutil.copy2(source, target)
        selected[f'media/{name}'] = {'bytes': target.stat().st_size, 'sha256': digest(target)}
    for name in SOURCE_CODE:
        source = ROOT / name
        target = DEST / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        selected[f'code/{name}'] = {'bytes': target.stat().st_size, 'sha256': digest(target)}
    (DEST / 'REVIEW_BUNDLE_MANIFEST.json').write_text(json.dumps({'files': selected}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'catalog_videos': len(all_videos), 'catalog_bytes': catalog['total_bytes'], 'review_files': len(selected), 'review_bytes': sum(v['bytes'] for v in selected.values())}))


if __name__ == '__main__':
    main()
