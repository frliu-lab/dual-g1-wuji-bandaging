"""Summarize all A-left moving-forearm WBC sweeps with a strict contact gate."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
files=sorted(OUT.glob('dynamic_palm_support_wbc_comparison*.json'))
rows=[]
for path in files:
    report=json.loads(path.read_text(encoding='utf8'))
    baseline=report['conditions']['no_contact']
    test=report['conditions']['palm_candidate_path']
    impulses=test['a_left_impulse_by_hand_body_ns']
    total=sum(impulses.values())
    palm=impulses.get('g1_a/wuji_left/l_wrist',0.)
    mount=impulses.get('g1_a/wuji_left/l_mount',0.)
    index=sum(v for k,v in impulses.items() if '/l_index_' in k)
    last=test.get('final_half_second_impulse_by_hand_body_ns',{})
    last_total=sum(last.values())
    last_palm=last.get('g1_a/wuji_left/l_wrist',0.)
    last_fingers=sum(v for k,v in last.items() if '_finger_' in k or '/l_thumb_' in k)
    reduction=baseline['final_half_second_mean_abs_elbow_rad']-test['final_half_second_mean_abs_elbow_rad']
    gate={'elbow_reduction_at_least_0p05_rad':reduction>=.05,
          'matched_full_duration':baseline['simulated_time_s']>=3.79 and test['simulated_time_s']>=3.79,
          'baseline_collision_free':baseline['deep_nonfloor_robot_contact_count']==0 and baseline['finite_state'] and not baseline['errors'],
          'final_contact_at_least_90_percent':test['final_half_second_contact_fraction']>=.9,
          'last_half_palm_impulse_majority':last_palm/max(last_total,1e-9)>=.7,
          'last_half_all_finger_impulse_below_10_percent':last_fingers/max(last_total,1e-9)<.1,
          'peak_force_below_20_n':test['peak_a_left_forearm_force_n']<20.,
          'no_forearm_overlift_beyond_zero_rad':test['final_elbow_rad']>=0.,
          'both_robots_foot_supported':test['foot_support_fraction']>=.99,
          'no_deep_nonfloor_robot_contact':test['deep_nonfloor_robot_contact_count']==0,
          'finite_no_error':test['finite_state'] and not test['errors']}
    rows.append({'file':path.name,'reach_fraction':report.get('reach_fraction',1.),
        'index_extra_rad':report.get('index_extra_rad',0.),'segment_s':report.get('segment_s',1.2),
        'forearm_follow_gain':report.get('forearm_follow_gain',0.),
        'easing':report.get('easing','cubic'),
        'abort_on_support_fault':report.get('abort_on_support_fault',False),
        'abort_mode':report.get('abort_mode'),
        'follow_start_fraction':report.get('follow_start_fraction',1.),
        'force_feedback_gain':report.get('force_feedback_gain',0.),
        'forearm_mass_kg':report.get('forearm_mass_kg',.8),
        'wbc_forearm_force_cap_n':report.get('wbc_forearm_force_cap_n'),
        'wbc_finger_force_cap_n':report.get('wbc_finger_force_cap_n'),
        'candidate_root_x_m':report.get('candidate_root_x_m',-.3),
        'candidate_initial_elbow_rad':report.get('candidate_initial_elbow_rad',.25),
        'armrest_variant':report.get('armrest_variant',False),
        'forearm_solref_time_s':report.get('forearm_solref_time_s'),
        'elbow_reduction_rad':reduction,'palm_impulse_share':palm/max(total,1e-9),
        'mount_impulse_share':mount/max(total,1e-9),'index_impulse_share':index/max(total,1e-9),
        'last_half_palm_impulse_share':last_palm/max(last_total,1e-9),
        'last_half_all_finger_impulse_share':last_fingers/max(last_total,1e-9),
        'peak_forearm_force_n':test['peak_a_left_forearm_force_n'],
        'online_support_valid_fraction':test.get('online_support_valid_fraction'),
        'final_half_second_mean_force_n':test['final_half_second_mean_force_n'],
        'gate':gate,'passes_all':all(gate.values())})
summary={'scope':'Conditional simulation-only sweep; thresholds are provisional research screens, not clinical or hardware safety limits',
         'candidate_count':len(rows),'passing_count':sum(r['passes_all'] for r in rows),
         'rows':rows}
(OUT/'dynamic_palm_support_sweep_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps({'candidate_count':summary['candidate_count'],'passing_count':summary['passing_count'],
                  'best_elbow_reduction_rad':max(r['elbow_reduction_rad'] for r in rows)}))
