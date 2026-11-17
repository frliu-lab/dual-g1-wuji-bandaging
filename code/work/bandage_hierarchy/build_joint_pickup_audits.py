"""Write compact public audits for the large local joint rollout traces."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
RUNS={'hold_17p7':'65e2b2086889','pickup_10mm':'ff6fdc58cfbd',
      'pickup_20mm':'947858479e10','pickup_30mm':'c4041b6d7308'}
for label,token in RUNS.items():
    source=OUT/f'joint_support_pinch_staged_wbc_report_spool_settled_handle_{token}.json'
    content=source.read_bytes()
    report=json.loads(content)
    audit={k:v for k,v in report.items() if k!='trace'}
    audit['source_local_report']=source.name
    audit['source_sha256']=hashlib.sha256(content).hexdigest()
    audit['source_bytes']=len(content)
    audit['late_trace_10hz']=[{k:frame[k] for k in
        ('time_s','elbow_rad','support_valid','support_force_n',
         'support_impulse_by_body_ns','thumb_force_n','index_force_n',
         'b_thumb_force_n','b_index_force_n','b_opposed')}
        for frame in report['trace'][-100::5]]
    dest=OUT/f'joint_handle_{label}_audit.json'
    dest.write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf8')
    print(label,dest.name,dest.stat().st_size)
pad_source=OUT/'joint_dressing_pickup_checkpoint_lift0.02_force0.7_iclose0_ipre0_iw10.json'
pad_content=pad_source.read_bytes()
pad=json.loads(pad_content)
pad_audit={k:v for k,v in pad.items() if k!='trace'}
pad_audit['source_local_report']=pad_source.name
pad_audit['source_sha256']=hashlib.sha256(pad_content).hexdigest()
pad_audit['source_bytes']=len(pad_content)
pad_audit['index_contact_timeline_10hz']=[{
    'time_s':r['time_s'],'pad_lift_m':r['pad_lift_m'],
    'index_contact_bodies':[c['bodies'][1] for c in r['pad_contacts'] if 'index_finger' in c['bodies'][1]],
    'stage_contact_count':sum(c['geoms'][0] in ('dressing_stage','equipment_tray') for c in r['pad_contacts'])}
    for r in pad['trace'][::5]]
dest=OUT/'joint_dressing_pickup_failure_audit.json'
dest.write_text(json.dumps(pad_audit,indent=2),encoding='utf8')
print('dressing_pickup_failure',dest.name,dest.stat().st_size)
