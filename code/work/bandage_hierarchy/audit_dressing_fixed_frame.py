"""Audit frozen digit-frame follow experiments after the dressing pickup."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
PREFIX='joint_dressing_pickup_checkpoint_lift0.04_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_duration4_'
SUFFIX='_motionend2.5_smooth_xshift-0.075_syaw-0.22_at1.5_palmshift0,0,0.02_from75fdd3a3b4b1.json'
results={}
for when in (2.3,2.4,2.5,2.7):
    name=PREFIX+f'fixedframe{when:g}'+SUFFIX
    source=OUT/name;content=source.read_bytes();r=json.loads(content)
    results[f'{when:g}s']={'source_local_report':name,'sha256':hashlib.sha256(content).hexdigest(),
                           'completed':r['completed'],'last_time_s':r['trace'][-1]['time_s'],
                           'last_pad_lift_m':r['trace'][-1]['pad_lift_m'],
                           'last_support_palm_share':r['final_support_quality']['window_palm_impulse_share'],
                           'errors':r['errors']}
audit={'scope':'Diagnostic: freeze fingertip target frame after pickup to resist dressing rotation',
       'results':results,
       'interpretation':'Freezing earlier delayed slip in some trials but none passed a 1.5 s sustained grasp and support gate. Existing 0.5 s pass remains the validated bound.'}
dest=OUT/'joint_dressing_fixed_digit_frame_failure_audit.json'
dest.write_text(json.dumps(audit,indent=2),encoding='utf8')
print(dest,dest.stat().st_size)
