"""Kinematic collision audit to a static free-edge grasp candidate."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5.xml'))
data=mujoco.MjData(model)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
q0=data.qpos.copy();q0[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
screen=json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
row=screen['results'][0]
names=screen['joint_names']
qa=np.array([model.joint(name).qposadr[0] for name in names])
start=q0[qa].copy();goal=np.asarray(row['joint_targets_rad'])
arm=np.array([name.startswith('g1_b/right_') for name in names])
routes=[]
for mode in ('synchronized','fingers_first','arm_first'):
    worst={'penetration_m':0.,'progress':0.,'bodies':[],'is_flex':False}
    deep_samples=0
    for t in np.linspace(0.,1.,101):
        if mode=='synchronized':arm_u=finger_u=t
        elif mode=='fingers_first':finger_u=min(1.,2*t);arm_u=max(0.,2*t-1.)
        else:arm_u=min(1.,2*t);finger_u=max(0.,2*t-1.)
        u=np.where(arm,arm_u,finger_u)
        data.qpos[:]=q0;data.qpos[qa]=start+u*(goal-start)
        mujoco.mj_forward(model,data)
        sample_max=0.
        for c in data.contact:
            if c.dist>=0:continue
            geoms=[int(g) for g in c.geom if g>=0]
            bodies=[model.body(model.geom_bodyid[g]).name for g in geoms]
            if not any(name.startswith('g1_b/') for name in bodies) or 'floor' in [model.geom(g).name for g in geoms]:
                continue
            pen=-float(c.dist)
            sample_max=max(sample_max,pen)
            if pen>worst['penetration_m']:
                worst={'penetration_m':pen,'progress':float(t),'bodies':bodies,
                       'is_flex':bool(any(f>=0 for f in c.flex))}
        deep_samples+=sample_max>=.003
    routes.append({'route':mode,'worst':worst,'deep_sample_count':deep_samples,
                   'collision_screen_pass':deep_samples==0})
report={'scope':'Static interpolation of B-right joints with flexible strip frozen; no dynamic path or grasp force',
        'source':screen['scope'],'routes':routes}
(OUT/'joint_b_flex_free_edge_path_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
