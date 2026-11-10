"""Compare reproducible side-pinch feedback pilots against explicit provisional gates."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
paths=[OUT/'pad_side_pinch_attempt_2_50hz_0.6s.json']
paths += [p for p in OUT.glob('pad_side_pinch_balance_*_50hz_0.6s.json')
          if '_trace_' not in p.name and p.name!='pad_side_pinch_balance_2_50hz_0.6s.json']
cases=[]
for path in sorted(paths):
    d=json.loads(path.read_text(encoding='utf8'))
    row={'file':path.name,
         'path_index':d['side_pinch_path_index'],
         'force_balance_gain_m_per_n':d.get('force_balance_gain_m_per_n',0.),
         'pad_relative_gain':d.get('pad_relative_gain',0.),
         'hold_s':d.get('hold_s',1.),
         'palm_rotation_weight':d.get('palm_rotation_weight',10.),
         'wrist_roll_offset_rad':d.get('wrist_roll_offset_rad',0.),
         'wrist_roll_late_offset_rad':d.get('wrist_roll_late_offset_rad',0.),
         'foot_support_fraction':d['both_robot_foot_support_fraction'],
         'deep_robot_contact_samples':d['deep_nonfloor_robot_contact_samples'],
         'max_pad_displacement_m':d['max_pad_center_displacement_m'],
         'longest_opposed_contact_s':d['longest_continuous_opposed_contact_s'],
         'opposed_final_half_second_fraction':d['opposed_contact_fraction_final_half_second'],
         'peak_thumb_force_n':d['peak_digit_normal_force_n']['thumb'],
         'peak_index_force_n':d['peak_digit_normal_force_n']['index'],
         'index_goal_error_m':d['final_fingertip_position_error_m']['g1_a/wuji_right/r_index_finger_tip_sensor_frame'],
         'thumb_goal_error_m':d['final_thumb_goal_error_m']}
    row['provisional_contact_gate_pass']=(row['foot_support_fraction']>=.999 and
        row['deep_robot_contact_samples']==0 and row['max_pad_displacement_m']<.005 and
        row['longest_opposed_contact_s']>=2. and row['opposed_final_half_second_fraction']>=.9)
    cases.append(row)
report={'scope':'Simulation-only comparison on an isolated smaller-pad variant; no lift, cloth or complete bandaging',
        'provisional_contact_gate':{'foot_support_fraction_min':.999,
                                    'deep_contact_samples_max':0,
                                    'max_pad_displacement_m_exclusive':.005,
                                    'longest_opposed_contact_s_min':2.,
                                    'opposed_final_half_second_fraction_min':.9},
        'cases':cases,
        'passed_cases':sum(x['provisional_contact_gate_pass'] for x in cases)}
(OUT/'pad_side_pinch_feedback_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'cases':len(cases),'passed':report['passed_cases'],
                  'best_continuous_s':max(x['longest_opposed_contact_s'] for x in cases),
                  'best_final_half_second_fraction':max(x['opposed_final_half_second_fraction'] for x in cases)}))
