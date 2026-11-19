"""Verify one full second of simultaneous dressing grip and forearm support."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
STEM='joint_dressing_pickup_checkpoint_lift0.04_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_duration3.5_fixedframe2.2_motionend2.5_smooth_xshift-0.075_syaw-0.22_at1.5_palmshift0,0,0.023_palmshift20.01,0,0.02_at3_from75fdd3a3b4b1'
source=OUT/f'{STEM}.json';content=source.read_bytes();report=json.loads(content)
hold=[r for r in report['trace'] if r['time_s']>=2.5]
def force(row,digit):
    return sum(c['normal_force_n'] for c in row['pad_contacts']
               if any(f'g1_a/wuji_right/r_{digit}' in b for b in c['bodies']))
def stage(row):
    return [c for c in row['pad_contacts'] if any(g in ('dressing_stage','equipment_tray') for g in c['geoms'])]
thumb=[force(r,'thumb') for r in hold];index=[force(r,'index_finger') for r in hold]
audit={'scope':'Checkpoint continuation simulation: 1.0 s dressing hold after pickup; no wrapping, no hardware validation',
       'source_local_report':source.name,'source_sha256':hashlib.sha256(content).hexdigest(),
       'source_bytes':len(content),'completed':report['completed'],'errors':report['errors'],
       'hold_duration_s':1.,'hold_control_steps':len(hold),
       'hold_support_valid_steps':sum(r['support_valid'] for r in hold),
       'hold_stage_contact_steps':sum(bool(stage(r)) for r in hold),
       'thumb_force_range_n':[min(thumb),max(thumb)],
       'index_force_range_n':[min(index),max(index)],
       'pad_lift_range_m':[min(r['pad_lift_m'] for r in hold),max(r['pad_lift_m'] for r in hold)],
       'max_spool_shift_m':max(r['spool_shift_m'] for r in report['trace']),
       'max_pad_lateral_shift_m':report['max_pad_lateral_shift_m'],
       'final_support_quality':report['final_support_quality'],
       'all_feet_supported':report['all_feet_supported'],
       'any_deep_robot_contact':report['any_deep_robot_contact'],
       'pass':bool(report['completed'] and len(hold)==50 and all(r['support_valid'] for r in hold)
                   and not any(stage(r) for r in hold)
                   and min(thumb)>.05 and max(thumb)<2.
                   and min(index)>.05 and max(index)<2.
                   and min(r['pad_lift_m'] for r in hold)>.025
                   and max(r['spool_shift_m'] for r in report['trace'])<.005
                   and report['all_feet_supported'] and not report['any_deep_robot_contact']),
       'interpretation':'Conditional simulation subskill. A separate 4 s continuation loses grip near 3.76 s, so do not extrapolate to longer holding.'}
dest=OUT/'joint_dressing_one_second_hold_pass_audit.json'
dest.write_text(json.dumps(audit,indent=2),encoding='utf8')
print(json.dumps(audit))
