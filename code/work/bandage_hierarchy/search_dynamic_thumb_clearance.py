"""Search a fixed B-thumb offset that clears spool contacts in saved pregrasp states."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
src=max((p for p in OUT.glob('*position1500*handle*yawm30*middle_retracted*clearspool*faultstop*bhanddiag*.json')
         if '_bthumbstaged_' not in p.name),key=lambda p:p.stat().st_mtime)
report=json.loads(src.read_text(encoding='utf8'))
frames=[f for f in report['trace'] if f['b_hand_diagnostic'] and 10.8<=f['time_s']<=11.78]
frames=frames[::max(1,len(frames)//12)]
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
for name in ('bandage_roll','bandage_core','bandage_handle_prototype'):
    model.geom_margin[model.geom(name).id]=.008
data=mujoco.MjData(model)
joints=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_b/wuji_left/l_thumb_')]
qa=np.array([model.jnt_qposadr[j] for j in joints])
base_q=np.array([f['b_hand_diagnostic']['qpos'] for f in frames])
tip=model.body('g1_b/wuji_left/l_thumb_tip_sensor_frame').id
reference_tips=[]
for q in base_q:
    data.qpos[:]=q;mujoco.mj_forward(model,data);reference_tips.append(data.xpos[tip].copy())
def evaluate(offset):
    minimum=.008;tip_error=0.;violations=0
    for q,ref in zip(base_q,reference_tips):
        data.qpos[:]=q;data.qpos[qa]+=offset;mujoco.mj_forward(model,data)
        tip_error=max(tip_error,float(np.linalg.norm(data.xpos[tip]-ref)))
        for c in data.contact:
            if c.geom1<0 or c.geom2<0:continue
            geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
            bodies=(model.body(model.geom_bodyid[c.geom1]).name,model.body(model.geom_bodyid[c.geom2]).name)
            if any(g in ('bandage_roll','bandage_core','bandage_handle_prototype') for g in geoms) and \
               any(b.startswith('g1_b/wuji_left/l_thumb_') for b in bodies):
                minimum=min(minimum,float(c.dist));violations+=int(c.dist<.003)
    return minimum,tip_error,violations
rng=np.random.default_rng(20261116)
candidates=[np.zeros(len(qa))]
for scale in (.1,.2,.35,.5):
    candidates.extend(rng.normal(0,scale,(250,len(qa))))
rows=[]
for offset in candidates:
    if np.any(base_q[:,qa]+offset<model.jnt_range[joints,0]+.02) or \
       np.any(base_q[:,qa]+offset>model.jnt_range[joints,1]-.02):continue
    clearance,tip_error,violations=evaluate(offset)
    rows.append({'offset_rad':offset.tolist(),'min_thumb_spool_distance_m':clearance,
                 'max_tip_shift_m':tip_error,'violating_contacts':violations})
rows.sort(key=lambda r:(r['violating_contacts'],max(0.,.003-r['min_thumb_spool_distance_m']),
                        r['max_tip_shift_m'],np.linalg.norm(r['offset_rad'])))
result={'scope':'Static replay of saved dynamic pregrasp qpos; candidate thumb offsets, no closed-loop success claim',
        'source':src.name,'sample_count':len(frames),'searched':len(rows),
        'joints':[model.joint(j).name for j in joints],
        'best':rows[:12]}
(OUT/'joint_spool_dynamic_thumb_clearance_search.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({'sample_count':len(frames),'searched':len(rows),'best':rows[:3]}))
