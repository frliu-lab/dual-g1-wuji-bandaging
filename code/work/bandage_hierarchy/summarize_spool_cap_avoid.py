"""Summarize B-left spool side-contact searches; preserve all failed candidates."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
paths=sorted(OUT.glob('joint_spool_grasp_static*dynamicwarm_jointwarm*capavoid_screen.json'))
rows=[]
for path in paths:
    record=json.loads(path.read_text(encoding='utf8'))
    rows.append({'file':path.name,'goal_z_m':record['goals_m'][0][2],
                 'seed_index':record['seed_index'],'seed_scale_rad':record.get('seed_scale_rad',.18),
                 'eligible':record['eligible_static_endpoint'],
                 'deep_contact_count':len(record['deep_contacts']),
                 'index_cap_contact_count':len(record['index_cap_contacts']),
                 'normal_cosine':record['normal_cosine'],
                 'forces_n':record['forces_n']})
report={'scope':'Static endcap-avoidance IK search around one dynamic joint terminal state; no closed-loop claim',
        'candidate_count':len(rows),'eligible_count':sum(row['eligible'] for row in rows),
        'candidates':rows}
(OUT/'joint_spool_cap_avoid_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'candidate_count':report['candidate_count'],'eligible_count':report['eligible_count']}))
