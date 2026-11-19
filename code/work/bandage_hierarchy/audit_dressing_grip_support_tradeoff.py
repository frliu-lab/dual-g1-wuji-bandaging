"""Audit 1.5 s dressing grip when forearm support degrades late."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
STEM='joint_dressing_pickup_checkpoint_lift0.04_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_duration4_fixedframe2_motionend2.5_smooth_xshift-0.075_syaw-0.22_at1.5_palmshift0,0,0.023_palmshift20.01,0,0.02_at3_from75fdd3a3b4b1'
source=OUT/f'{STEM}.json';content=source.read_bytes();r=json.loads(content)
hold=[row for row in r['trace'] if row['time_s']>=2.5]
def force(row,digit):
    return sum(c['normal_force_n'] for c in row['pad_contacts']
               if any(f'g1_a/wuji_right/r_{digit}' in b for b in c['bodies']))
thumb=[force(row,'thumb') for row in hold];index=[force(row,'index_finger') for row in hold]
audit={'scope':'Diagnostic 1.5 s dressing grip with simultaneous forearm support gate',
       'source_local_report':source.name,'source_sha256':hashlib.sha256(content).hexdigest(),
       'completed':r['completed'],'hold_control_steps':len(hold),
       'grip_contact_steps':sum(force(row,'thumb')>.05 and force(row,'index_finger')>.05 for row in hold),
       'support_valid_steps':sum(row['support_valid'] for row in hold),
       'thumb_force_range_n':[min(thumb),max(thumb)],'index_force_range_n':[min(index),max(index)],
       'pad_lift_range_m':[min(row['pad_lift_m'] for row in hold),max(row['pad_lift_m'] for row in hold)],
       'final_support_palm_share':r['final_support_quality']['window_palm_impulse_share'],
       'interpretation':'Fingertip grip survives 1.5 s; forearm support fails its 90% palm-share gate in the late hold. No combined pass.'}
dest=OUT/'joint_dressing_grip_support_tradeoff_audit.json'
dest.write_text(json.dumps(audit,indent=2),encoding='utf8')
print(json.dumps(audit))
