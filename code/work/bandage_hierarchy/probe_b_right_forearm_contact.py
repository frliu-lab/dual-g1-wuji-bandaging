"""Static B-right hand contact scan at the validated planted stance."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
screen=json.loads((OUT/'joint_b_reposition_static_screen.json').read_text())
path=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
model=mujoco.MjModel.from_xml_path(str(OUT/screen['scene']))
data=mujoco.MjData(model)
q=np.asarray(path['samples'][-1]['qpos'])
hand=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_b/wuji_right/r_')
      and 'abd' not in model.joint(j).name]
rows=[]
for flex in np.linspace(0,.9,19):
    data.qpos[:]=q
    for j in hand:
        data.qpos[model.jnt_qposadr[j]]=np.clip(flex,model.jnt_range[j,0]+.01,model.jnt_range[j,1]-.01)
    mujoco.mj_forward(model,data)
    hits=[];deep_other=[]
    for c in data.contact:
        geoms=[model.geom(g).name for g in c.geom]
        bodies=[model.body(model.geom_bodyid[g]).name for g in c.geom]
        if 'training_forearm_dynamic' in bodies and any(n.startswith('g1_b/wuji_right/') for n in bodies):
            hits.append({'bodies':bodies,'distance_m':float(c.dist)})
        elif c.dist<-.003 and 'floor' not in geoms and any(n.startswith('g1_b/') for n in bodies):
            deep_other.append({'bodies':bodies,'distance_m':float(c.dist)})
    rows.append({'uniform_flex_rad':float(flex),
                 'b_right_forearm_contact_count':sum(h['distance_m']<.001 for h in hits),
                 'b_right_forearm_max_penetration_m':max((-h['distance_m'] for h in hits),default=0.),
                 'b_right_forearm_contact_bodies':sorted(set(n for h in hits for n in h['bodies'] if n.startswith('g1_b/'))),
                 'deep_other_contact_count':len(deep_other)})
report={'scope':'Single-state geometry scan only; uniform finger flex is not a grasp controller',
        'rows':rows}
(OUT/'joint_b_right_forearm_flex_contact_scan.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
