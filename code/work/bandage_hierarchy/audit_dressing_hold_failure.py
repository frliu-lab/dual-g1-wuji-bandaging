"""Summarize why the current A-right two-finger pickup cannot be called a hold."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
STEMS={
    'maintain_hard_upward_force':'joint_dressing_pickup_checkpoint_lift0.04_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_motionend2.5_xshift-0.075_syaw-0.22_at1.5_from75fdd3a3b4b1',
    'release_hard_force':'joint_dressing_pickup_checkpoint_lift0.04_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_motionend2.5_forceend2.5_xshift-0.075_syaw-0.22_at1.5_from75fdd3a3b4b1',
    'release_and_raise_soft_grip':'joint_dressing_pickup_checkpoint_lift0.04_force1.2_iclose0_ipre-1_side0_iw10_up0.12_start1.65_motionend2.5_forceend2.5_xshift-0.075_syaw-0.22_at1.5_from75fdd3a3b4b1',
}
def digit_force(row,label):
    return sum(c['normal_force_n'] for c in row['pad_contacts']
               if any(f'g1_a/wuji_right/r_{label}' in b for b in c['bodies']))
results={}
for label,stem in STEMS.items():
    source=OUT/f'{stem}.json';content=source.read_bytes();r=json.loads(content)
    final=r['trace'][-1]
    results[label]={'source_local_report':source.name,'source_sha256':hashlib.sha256(content).hexdigest(),
                    'source_bytes':len(content),'completed':r['completed'],'errors':r['errors'],
                    'last_time_s':final['time_s'],'final_pad_lift_m':final['pad_lift_m'],
                    'final_thumb_force_n':digit_force(final,'thumb'),
                    'final_index_force_n':digit_force(final,'index_finger'),
                    'final_support_palm_share':r['final_support_quality']['window_palm_impulse_share']}
audit={'scope':'Failure audit for sustained dressing hold after pickup',
       'conclusion':'Neither hard upward force nor its release produced a valid 0.5 s sustained grasp. The current pickup demo remains a transient pickup, not a stable hold.',
       'results':results}
dest=OUT/'joint_dressing_sustained_hold_failure_audit.json'
dest.write_text(json.dumps(audit,indent=2),encoding='utf8')
print(dest,dest.stat().st_size)
