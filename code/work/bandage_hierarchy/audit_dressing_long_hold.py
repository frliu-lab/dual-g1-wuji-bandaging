"""Compact evidence for failed 1.5 s dressing hold extension."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
STEM='joint_dressing_pickup_checkpoint_lift0.04_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_motionend2.5_smooth_xshift-0.075_syaw-0.22_at1.5_palmshift0,0,0.02_from75fdd3a3b4b1_duration4'
source=OUT/f'{STEM}.json';content=source.read_bytes();report=json.loads(content)
trace=report['trace']
def force(row,digit):
    return sum(c['normal_force_n'] for c in row['pad_contacts']
               if any(f'g1_a/wuji_right/r_{digit}' in b for b in c['bodies']))
audit={'scope':'Attempt to extend successful 0.5 s dressing hold to 1.5 s',
       'source_local_report':source.name,'source_sha256':hashlib.sha256(content).hexdigest(),
       'source_bytes':len(content),'completed':report['completed'],'errors':report['errors'],
       'last_time_s':trace[-1]['time_s'],
       'last_pad_lift_m':trace[-1]['pad_lift_m'],
       'last_support_palm_share':report['final_support_quality']['window_palm_impulse_share'],
       'contact_timeline_50hz':[{'time_s':r['time_s'],'thumb_force_n':force(r,'thumb'),
                                 'index_force_n':force(r,'index_finger'),
                                 'pad_lift_m':r['pad_lift_m']} for r in trace if r['time_s']>=2.9],
       'interpretation':'Grip contacts collapse near 3.06 s. The 0.5 s pass cannot be extrapolated to a longer hold.'}
dest=OUT/'joint_dressing_long_hold_failure_audit.json'
dest.write_text(json.dumps(audit,indent=2),encoding='utf8')
print(dest,dest.stat().st_size)
