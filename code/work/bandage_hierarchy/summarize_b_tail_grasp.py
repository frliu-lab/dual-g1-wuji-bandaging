"""Strict gate for the five compiled-geometry two-fingertip tail seeds."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
rows=[]
def gate(trial):
    return (trial['error'] is None and trial['duration_s']>=.499 and
            trial['contact_fractions']['two_feet']>=.99 and
            trial['contact_fractions']['opposing']>=.9 and
            trial['max_tail_center_drift_m']<=.005 and
            trial['max_b_nonfloor_penetration_m']<.003)
for rank in range(5):
    name=f'joint_b_tail_dynamic_proxy_grasp_pilot_7mm_tip_pinch_rank{rank}.json'
    trial=json.loads((OUT/name).read_text())
    opposing=trial['contact_fractions']['opposing']
    drift=trial['max_tail_center_drift_m']
    penetration=trial['max_b_nonfloor_penetration_m']
    passed=gate(trial)
    rows.append({'rank':rank,'source':name,'duration_s':trial['duration_s'],
                 'error':trial['error'],'opposing_contact_fraction':opposing,
                 'tail_center_drift_m':drift,'max_b_nonfloor_penetration_m':penetration,
                 'two_foot_support_fraction':trial['contact_fractions']['two_feet'],
                 'strict_proxy_grasp_pass':passed})
feedback=[]
for label,suffix in (('gap_feedback','_gap'),('gap_and_1n_force','_force1n_gap'),
                     ('pose_tip_servo','_tip_servo_1mm_w1000_pose')):
    name=f'joint_b_tail_dynamic_proxy_grasp_pilot_7mm_tip_pinch_rank0{suffix}.json'
    trial=json.loads((OUT/name).read_text())
    feedback.append({'label':label,'source':name,'duration_s':trial['duration_s'],
                     'error':trial['error'],
                     'opposing_contact_fraction':trial['contact_fractions']['opposing'],
                     'tail_center_drift_m':trial['max_tail_center_drift_m'],
                     'strict_proxy_grasp_pass':gate(trial)})
report={'scope':'Rigid tail on rest with massless tether; research gate only, no flexible bandage or hardware claim',
        'thresholds':{'duration_s_min':.499,'two_foot_support_fraction_min':.99,
                      'opposing_contact_fraction_min':.9,'tail_center_drift_m_max':.005,
                      'max_b_nonfloor_penetration_m':.003,'wbc_error':None},
        'passed_count':sum(r['strict_proxy_grasp_pass'] for r in rows),
        'trial_count':len(rows),'trials':rows,'feedback_trials':feedback}
(OUT/'joint_b_tail_tip_pinch_grasp_gate.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
