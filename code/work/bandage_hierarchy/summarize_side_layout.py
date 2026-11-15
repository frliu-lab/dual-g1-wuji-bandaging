"""Compare prepositioned B side-grip layouts using actual closed-loop contacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
paths=sorted(OUT.glob('joint_support_pinch_staged_wbc_report*settled_side*.json'))
rows=[]
for path in paths:
    d=json.loads(path.read_text(encoding='utf8'))
    support=d['online_support_quality_final']['valid']
    pad=d['longest_opposed_contact_s']>=1.
    spool=(d['longest_b_left_opposed_spool_contact_s']>=1.
           and d['peak_b_left_thumb_force_n']<=2.
           and d['peak_b_left_index_force_n']<=2.
           and d['max_spool_center_displacement_m']<=.005)
    balance=d['foot_support_fraction']>=.99 and d['deep_nonfloor_robot_contact_control_steps']==0
    row={'file':path.name,'duration_s':d['simulated_time_s'],
         'support_final_valid':support,'pad_opposed_s':d['longest_opposed_contact_s'],
         'spool_opposed_s':d['longest_b_left_opposed_spool_contact_s'],
         'spool_peak_thumb_n':d['peak_b_left_thumb_force_n'],
         'spool_peak_index_n':d['peak_b_left_index_force_n'],
         'spool_max_displacement_m':d['max_spool_center_displacement_m'],
         'deep_contact_steps':d['deep_nonfloor_robot_contact_control_steps'],
         'support_and_pad':bool(support and pad and balance and not d['errors']),
         'joint_gate':bool(support and pad and spool and balance and not d['errors'])}
    rows.append(row)
report={'scope':'Provisional simulated side-layout screens, not hardware or bandaging qualification',
        'gates':{'support_final':True,'pad_opposed_s':1.,'spool_opposed_s':1.,
                 'spool_peak_each_digit_n':2.,'spool_max_displacement_m':.005,
                 'foot_support_fraction':.99,'deep_contact_steps':0},
        'candidate_count':len(rows),'support_and_pad_count':sum(r['support_and_pad'] for r in rows),
        'joint_gate_count':sum(r['joint_gate'] for r in rows),'candidates':rows}
(OUT/'joint_side_layout_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'candidate_count':report['candidate_count'],
                  'support_and_pad_count':report['support_and_pad_count'],
                  'joint_gate_count':report['joint_gate_count']}))
