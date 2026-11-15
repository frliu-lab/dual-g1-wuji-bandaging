"""Summarize fixed B-root x/y grid under strict static contact screens."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
rows=[]
for y in (.05,.1,.15):
    for x in (.5,.55,.6):
        for z in (.72,.725,.73):
            stem='joint_spool_grasp_static'+('' if z==.72 else '_z'+str(z).replace('.','p'))
            stem+='_closure0p004_root'+str(x).replace('.','p')
            stem+='_rooty'+str(y).replace('.','p')+'_spoolz0p702_screen.json'
            path=OUT/stem
            d=json.loads(path.read_text(encoding='utf8'))
            rows.append({'file':path.name,'b_root_x_m':x,'b_root_y_m':y,'target_z_m':z,
                         'strict_static_endpoint':d['strict_static_endpoint'],
                         'max_tip_goal_error_m':max(d['tip_errors_m']),
                         'normal_cosine':d['normal_cosine'],
                         'deep_contacts':len(d['deep_contacts']),
                         'forces_n':d['forces_n']})
report={'scope':'27 static prepositioned B-layout probes; no B locomotion, dynamic grasp or bandaging claim',
        'candidate_count':len(rows),
        'strict_static_count':sum(r['strict_static_endpoint'] for r in rows),
        'candidates':rows}
(OUT/'joint_spool_layout_grid_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'candidate_count':report['candidate_count'],
                  'strict_static_count':report['strict_static_count']}))
