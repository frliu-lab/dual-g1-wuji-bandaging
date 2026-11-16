"""Reclassify saved handle contacts by radial side versus endcap geometry."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
rows=[]
for path in sorted(OUT.glob('joint_support_pinch_staged_wbc_report*settled_handle*.json')):
    d=json.loads(path.read_text(encoding='utf8'))
    side_both=0;side_opposed=0;cap=0;off_target=0;min_side_cos=None
    for frame in d['trace']:
        all_details=frame.get('b_contact_details',[])
        details=[x for x in all_details if x.get('spool_geom')=='bandage_handle_prototype']
        def is_side(contact):
            axis=contact.get('handle_axis_world')
            normal=np.asarray(contact['spool_to_finger_normal'])
            return abs(float(normal@np.asarray(axis if axis is not None else [0.,0.,1.])))<.5
        if any(not is_side(x) for x in details):cap+=1
        if any(x.get('spool_geom')!='bandage_handle_prototype' for x in all_details):off_target+=1
        forces={'thumb':0.,'index':0.};normals={digit:np.zeros(3) for digit in forces}
        for c in details:
            if not is_side(c):continue
            digit=c['digit'];f=c['force_n']
            forces[digit]+=f;normals[digit]+=f*np.asarray(c['spool_to_finger_normal'])
        if min(forces.values())<.05:continue
        side_both+=1
        a,b=normals.values();cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
        min_side_cos=cos if min_side_cos is None else min(min_side_cos,cos)
        side_opposed+=int(cos<-.5)
    rows.append({'file':path.name,'simulated_time_s':d['simulated_time_s'],
                 'completed':not bool(d.get('errors')) and d['simulated_time_s']>=d.get('planned_duration_s',0)-.03,
                 'errors':d.get('errors',[]),
                 'general_b_opposed_s':d['longest_b_left_opposed_spool_contact_s'],
                 'side_both_control_steps':side_both,'side_opposed_control_steps':side_opposed,
                 'min_side_normal_cosine':min_side_cos,'cap_contact_control_steps_when_both_digits_touched':cap,
                 'roll_or_core_contact_control_steps_when_both_digits_touched':off_target})
report={'scope':'Offline geometric re-audit of saved prototype-handle contact traces; 50 Hz control frames; no new simulation',
        'run_count':len(rows),'runs_with_side_opposed':sum(r['side_opposed_control_steps']>0 for r in rows),
        'completed_run_count':sum(r['completed'] for r in rows),
        'runs':rows}
(OUT/'joint_spool_handle_side_contact_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'run_count':report['run_count'],'runs_with_side_opposed':report['runs_with_side_opposed']}))
