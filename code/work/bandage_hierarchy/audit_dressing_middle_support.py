"""Compare static middle-finger contacts with dynamic hold trials."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
names={
    'middle_from_start':'joint_dressing_pickup_checkpoint_lift0.04_force0.7_iclose0_ipre-1_side-1_iw10_up0.12_start1.65_middle0_motionend2.5_xshift-0.075_syaw-0.22_at1.5_from75fdd3a3b4b1.json',
    'middle_added_late':'joint_dressing_pickup_checkpoint_lift0.04_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_middle0_middleafter2.2_motionend2.5_xshift-0.075_syaw-0.22_at1.5_from75fdd3a3b4b1.json',
}
results={}
for label,name in names.items():
    source=OUT/name;content=source.read_bytes();report=json.loads(content)
    final=report['trace'][-1]
    results[label]={'source_local_report':name,'sha256':hashlib.sha256(content).hexdigest(),
                    'completed':report['completed'],'errors':report['errors'],
                    'last_time_s':final['time_s'],'final_pad_lift_m':final['pad_lift_m'],
                    'final_support_palm_share':report['final_support_quality']['window_palm_impulse_share']}
static=OUT/'joint_dressing_middle_support_screen.json'
summary=json.loads(static.read_text(encoding='utf8'))
audit={'scope':'Static versus dynamic A-right middle-finger support comparison',
       'static_screened':summary['screened'],'static_collision_free_contact_count':summary['collision_free_contact_count'],
       'static_source':static.name,'static_sha256':hashlib.sha256(static.read_bytes()).hexdigest(),
       'dynamic_results':results,
       'conclusion':'Static third-finger candidates did not maintain the dynamic thumb/index grasp; no sustained hold pass.'}
dest=OUT/'joint_dressing_middle_support_failure_audit.json'
dest.write_text(json.dumps(audit,indent=2),encoding='utf8')
print(dest,dest.stat().st_size)
