"""Check whether finger closure can be delayed during the joint approach path."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_support_pinch_variant.xml'))
data=mujoco.MjData(model)
left=np.load(OUT/'dynamic_palm_support_rrt.npz')['qpos']
right=np.load(OUT/'joint_support_pinch_rrt.npz')['qpos']
fingers={digit:np.array([model.jnt_qposadr[j] for j in range(model.njnt)
                        if model.joint(j).name.startswith(f'g1_a/wuji_right/r_{digit}_')],dtype=int)
         for digit in ('thumb','index')}
results=[]
for mode,held in (('thumb_last',('thumb',)),('index_last',('index',)),
                  ('both_last',('thumb','index'))):
    staged=right.copy()
    for digit in held:
        staged[:,fingers[digit]]=right[0,fingers[digit]]
    staged=np.vstack([staged,right[-1]])
    deep=[]
    first_pad_touch={digit:None for digit in ('thumb','index')}
    sampled=0
    for segment,(start,end) in enumerate(zip(staged,staged[1:]),start=1):
        n=max(2,int(np.ceil(np.linalg.norm(end-start)/.04)))
        for k,u in enumerate(np.linspace(0,1,n+1)):
            data.qpos[:]=(1-u)*start+u*end
            mujoco.mj_forward(model,data)
            sampled+=1
            for c in data.contact:
                if c.geom1<0 or c.geom2<0:continue
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                        model.body(model.geom_bodyid[c.geom2]).name)
                if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
                    if len(deep)<10:deep.append({'segment':segment,'fraction':float(u),'bodies':bodies,
                                                 'distance_m':float(c.dist)})
                if 'dressing_pad_contact' in geoms:
                    for digit in first_pad_touch:
                        if first_pad_touch[digit] is None and any(b.startswith(f'g1_a/wuji_right/r_{digit}_') for b in bodies):
                            first_pad_touch[digit]={'segment':segment,'fraction':float(u)}
    safe=len(deep)==0
    results.append({'mode':mode,'dense_samples':sampled,'deep_contacts':deep,
                    'statically_collision_free':safe,'first_pad_touch':first_pad_touch})
    if safe:
        full=np.vstack([left,staged[1:]])
        np.savez_compressed(OUT/f'joint_support_pinch_staged_{mode}.npz',qpos=full)
report={'scope':'Static choreography screen only; moving-pad dynamics and WBC not included','results':results}
(OUT/'joint_support_pinch_staged_screen.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
