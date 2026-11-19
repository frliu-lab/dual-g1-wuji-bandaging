"""Verify the final 0.5 s of the dressing pickup checkpoint simulation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
SOURCE=OUT/'joint_dressing_pickup_checkpoint_lift0.04_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_motionend2.5_smooth_xshift-0.075_syaw-0.22_at1.5_palmshift0,0,0.02_from75fdd3a3b4b1.json'
content=SOURCE.read_bytes();report=json.loads(content)
trace=report['trace'];hold=[r for r in trace if r['time_s']>=2.5]
def digit_force(row,digit):
    return sum(c['normal_force_n'] for c in row['pad_contacts']
               if any(f'g1_a/wuji_right/r_{digit}' in b for b in c['bodies']))
def stage_contact(row):
    return [c for c in row['pad_contacts'] if any(g in ('dressing_stage','equipment_tray') for g in c['geoms'])]
thumb=[digit_force(r,'thumb') for r in hold]
index=[digit_force(r,'index_finger') for r in hold]
audit={'scope':'Checkpoint continuation: 0.5 s sustained dressing grasp after 2.5 s pickup; no full-controller route or wrapping claim',
       'source_local_report':SOURCE.name,'source_sha256':hashlib.sha256(content).hexdigest(),
       'source_bytes':len(content),'completed':report['completed'],'errors':report['errors'],
       'hold_duration_s':.5,'hold_control_steps':len(hold),
       'hold_support_valid_steps':sum(r['support_valid'] for r in hold),
       'hold_stage_contact_steps':sum(bool(stage_contact(r)) for r in hold),
       'hold_thumb_force_range_n':[min(thumb),max(thumb)],
       'hold_index_force_range_n':[min(index),max(index)],
       'hold_pad_lift_range_m':[min(r['pad_lift_m'] for r in hold),max(r['pad_lift_m'] for r in hold)],
       'final_support_quality':report['final_support_quality'],
       'max_spool_shift_m':max(r['spool_shift_m'] for r in trace),
       'all_feet_supported':report['all_feet_supported'],
       'any_deep_robot_contact':report['any_deep_robot_contact'],
       'pad_lateral_shift_max_m':report['max_pad_lateral_shift_m'],
       'pass':bool(report['completed'] and len(hold)>=24 and all(r['support_valid'] for r in hold)
                   and all(not stage_contact(r) for r in hold)
                   and min(thumb)>.05 and max(thumb)<2. and min(index)>.05 and max(index)<2.
                   and min(r['pad_lift_m'] for r in hold)>.025 and
                   report['all_feet_supported'] and not report['any_deep_robot_contact'] and
                   max(r['spool_shift_m'] for r in trace)<.005),
       'interpretation':'Pass is conditional simulation checkpoint subskill. Spool is grasped on tray, not lifted. No wrap or hardware proof.'}
dest=OUT/'joint_dressing_sustained_half_second_pass_audit.json'
dest.write_text(json.dumps(audit,indent=2),encoding='utf8')
print(json.dumps(audit))
