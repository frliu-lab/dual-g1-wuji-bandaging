"""Compact, reproducible audit of the checkpoint dressing pickup experiment."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
SOURCE=OUT/'joint_dressing_pickup_checkpoint_lift0.035_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_xshift-0.065_syaw-0.22_at1.5.json'
content=SOURCE.read_bytes()
report=json.loads(content)
trace=report['trace']
late=[row for row in trace if row['time_s']>=trace[-1]['time_s']-.3]
def stage_contacts(row):
    return [c for c in row['pad_contacts'] if any(g in ('dressing_stage','equipment_tray') for g in c['geoms'])]
def digit_force(row,label):
    return sum(c['normal_force_n'] for c in row['pad_contacts']
               if any(f'g1_a/wuji_right/r_{label}' in b for b in c['bodies']))
audit={k:v for k,v in report.items() if k not in ('trace','final_pad_contacts')}
audit.update({'source_local_report':SOURCE.name,'source_sha256':hashlib.sha256(content).hexdigest(),
              'source_bytes':len(content),'late_window_s':.3,
              'late_stage_contact_steps':sum(bool(stage_contacts(r)) for r in late),
              'late_stage_max_force_n':max((c['normal_force_n'] for r in late for c in stage_contacts(r)),default=0.),
              'late_stage_max_penetration_m':max((-c['distance_m'] for r in late for c in stage_contacts(r)),default=0.),
              'late_support_valid_steps':sum(r['support_valid'] for r in late),
              'late_steps':len(late),
              'max_spool_shift_m':max(r['spool_shift_m'] for r in trace),
              'late_thumb_peak_n':max(digit_force(r,'thumb') for r in late),
              'late_index_peak_n':max(digit_force(r,'index_finger') for r in late),
              'final_thumb_force_n':digit_force(trace[-1],'thumb'),
              'final_index_force_n':digit_force(trace[-1],'index_finger'),
              'late_trace_10hz':[{'time_s':r['time_s'],'pad_lift_m':r['pad_lift_m'],
                                  'stage_force_n':sum(c['normal_force_n'] for c in stage_contacts(r)),
                                  'support_palm_share':r['support_quality'].get('window_palm_impulse_share'),
                                  'thumb_force_n':digit_force(r,'thumb'),
                                  'index_force_n':digit_force(r,'index_finger')}
                                 for r in late[::5]],
              'interpretation':'Checkpoint pilot only. No full-route replay, wrap, tape fixation, or hardware test.'})
dest=OUT/'joint_dressing_side_pickup_checkpoint_audit.json'
dest.write_text(json.dumps(audit,indent=2),encoding='utf8')
print(dest, dest.stat().st_size)
