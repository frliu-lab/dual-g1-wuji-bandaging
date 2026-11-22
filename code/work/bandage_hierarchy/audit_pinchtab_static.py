"""Audit contact normals and penetration of the measured-gap cloth tab design."""
import json
import argparse
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--bulged',action='store_true')
args=parser.parse_args()
scene=OUT/('joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab_bulged.xml'
           if args.bulged else
           'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab.xml')
m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
q=d.qpos.copy();q[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
reach=json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
q[[m.joint(n).qposadr[0] for n in reach['joint_names']]]=reach['results'][0]['joint_targets_rad']
hand=json.loads((OUT/'joint_b_natural_pinch_workspace_screen.json').read_text())
candidate=next(row for row in hand['results'] if row['base']=='reach' and row['trial']==4)
q[[m.joint(n).qposadr[0] for n in hand['joint_names']]]=candidate['joint_targets_rad']
d.qpos[:]=q;mujoco.mj_forward(m,d)
contacts={'thumb':[],'index':[]};max_pen=0.;worst=[]
for c in d.contact:
    geoms=[int(g) for g in c.geom if g>=0]
    bodies=[m.body(m.geom_bodyid[g]).name for g in geoms]
    if c.dist<0 and any(n.startswith('g1_b/') for n in bodies) and 'floor' not in [m.geom(g).name for g in geoms]:
        pen=-float(c.dist);max_pen=max(max_pen,pen)
        worst.append({'bodies':bodies,'flex':bool(any(f>=0 for f in c.flex)),
                      'penetration_m':pen,'position_m':list(map(float,c.pos))})
    if not any(f>=0 for f in c.flex) or c.dist>=.003:continue
    for name,token in [('thumb','r_thumb_tip_sensor_frame'),('index','r_index_finger_tip_sensor_frame')]:
        if any(token in n for n in bodies):
            contacts[name].append({'dist_m':float(c.dist),'normal':list(map(float,c.frame[:3])),
                                   'position_m':list(map(float,c.pos))})
pairs=[{'normal_dot':float(np.dot(a['normal'],b['normal'])),
        'separation_m':float(np.linalg.norm(np.asarray(a['position_m'])-b['position_m']))}
       for a in contacts['thumb'] for b in contacts['index']]
report={'scope':'Static measured-gap flexible-tab audit; no grasp force or dynamic execution',
        'thumb_flex_contacts':len(contacts['thumb']),'index_flex_contacts':len(contacts['index']),
        'opposed_pair_count':sum(p['normal_dot']<-.5 for p in pairs),
        'max_b_nonfloor_penetration_m':max_pen,
        'static_contact_gate_pass':bool(any(p['normal_dot']<-.5 for p in pairs) and max_pen<.003),
        'nearest_contact_pair':min(pairs,key=lambda p:p['normal_dot']) if pairs else None,
        'contacts':contacts,'worst_contacts':sorted(worst,key=lambda x:x['penetration_m'],reverse=True)[:12]}
for label,pose in [('planted_seed',None),('reach_seed',reach['results'][0]['joint_targets_rad'])]:
    d.qpos[:]=d.qpos0 if hasattr(d,'qpos0') else m.qpos0
    d.qpos[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
    if pose is not None:
        d.qpos[[m.joint(n).qposadr[0] for n in reach['joint_names']]]=pose
    mujoco.mj_forward(m,d)
    pen=0.
    for c in d.contact:
        if c.dist>=0:continue
        geoms=[int(g) for g in c.geom if g>=0]
        bodies=[m.body(m.geom_bodyid[g]).name for g in geoms]
        if any(n.startswith('g1_b/') for n in bodies) and 'floor' not in [m.geom(g).name for g in geoms]:
            pen=max(pen,-float(c.dist))
    report[label+'_max_b_nonfloor_penetration_m']=pen
(OUT/('joint_b_pinchtab_bulged_static_contact_audit.json' if args.bulged else
      'joint_b_pinchtab_static_contact_audit.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('contacts','worst_contacts')}))
