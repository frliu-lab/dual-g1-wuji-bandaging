"""Check actual index-tip contact while preserving the static thumb-side pose."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5.xml'))
d=mujoco.MjData(m)
screen=json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
q0=d.qpos.copy();q0[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
qa=[m.joint(name).qposadr[0] for name in screen['joint_names']]
q0[qa]=screen['results'][0]['joint_targets_rad']
d.qpos[:]=q0;mujoco.mj_forward(m,d)
vertices=d.flexvert_xpos[:66].reshape(33,2,3)
centers=vertices.mean(axis=1)
tangent=centers[28]-centers[27]
width=vertices[28,1]-vertices[28,0]
normal=np.cross(tangent,width);normal/=np.linalg.norm(normal)
index_body=m.body('g1_b/wuji_right/r_index_finger_tip_sensor_frame').id
joint_names=[name for name in screen['joint_names'] if '/r_index_finger_' in name]
jids=[m.joint(name).id for name in joint_names]
qidx=np.array([m.jnt_qposadr[j] for j in jids])
lo=m.jnt_range[jids,0]+1e-4;hi=m.jnt_range[jids,1]-1e-4
start=q0[qidx]
rows=[]

def evaluate(x):
    d.qpos[:]=q0;d.qpos[qidx]=x;mujoco.mj_forward(m,d)
    tip=d.xpos[index_body].copy()
    max_pen=0.;digit={'thumb':[],'index':[]}
    for c in d.contact:
        if c.dist>=0:continue
        geoms=[int(g) for g in c.geom if g>=0]
        bodies=[m.body(m.geom_bodyid[g]).name for g in geoms]
        if any(n.startswith('g1_b/') for n in bodies) and 'floor' not in [m.geom(g).name for g in geoms]:
            max_pen=max(max_pen,-float(c.dist))
        if not any(f>=0 for f in c.flex):continue
        for key,token in [('thumb','r_thumb_tip_sensor_frame'),('index','r_index_finger_tip_sensor_frame')]:
            if any(token in n for n in bodies):digit[key].append((float(c.dist),c.frame[:3].copy()))
    opposing=any(np.dot(a[1],b[1])<-.5 for a in digit['thumb'] for b in digit['index'])
    return tip,max_pen,len(digit['thumb']),len(digit['index']),opposing

for offset_mm in (20,16,12,10,8,6,4):
    target=centers[28]-normal*offset_mm/1000
    def residual(x):
        tip,pen,*_=evaluate(x)
        return np.r_[.3*(tip-target),8*max(0.,pen-.003),.001*(x-start)]
    fit=least_squares(residual,np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=100)
    tip,pen,thumb,index,opposing=evaluate(fit.x)
    rows.append({'offset_mm':offset_mm,'optimizer_success':bool(fit.success),
                 'tip_target_error_m':float(np.linalg.norm(tip-target)),
                 'max_b_nonfloor_penetration_m':pen,
                 'thumb_tip_contact_count':thumb,'index_tip_contact_count':index,
                 'opposed_contact_normals':bool(opposing),
                 'joint_targets_rad':fit.x.tolist()})
report={'scope':'Fixed arm/thumb static index-joint closure; real contact geometry checked, no grasp force or dynamic claim',
        'joint_names':joint_names,'results':rows}
(OUT/'joint_b_flex_index_closure_screen.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='joint_targets_rad'} for r in rows]))
