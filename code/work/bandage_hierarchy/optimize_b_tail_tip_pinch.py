"""Jointly optimize B right arm and two digits around a fixed rigid tail proxy."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
screen=json.loads((OUT/'joint_b_reposition_static_screen.json').read_text())
path=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
q0=np.asarray(path['samples'][-1]['qpos'])
base_model=mujoco.MjModel.from_xml_path(str(OUT/screen['scene']))
base_data=mujoco.MjData(base_model)
base_data.qpos[:]=q0
for j in range(base_model.njnt):
    if base_model.joint(j).name.startswith('g1_b/wuji_right/r_index_') and 'abd' not in base_model.joint(j).name:
        base_data.qpos[base_model.jnt_qposadr[j]]=.325
mujoco.mj_forward(base_model,base_data)
base_thumb_bid=base_model.body('g1_b/wuji_right/r_thumb_tip_sensor_frame').id
base_index_bid=base_model.body('g1_b/wuji_right/r_index_finger_tip_sensor_frame').id
base_thumb_gid=next(i for i in range(base_model.ngeom) if base_model.geom_bodyid[i]==base_thumb_bid)
base_index_gid=next(i for i in range(base_model.ngeom) if base_model.geom_bodyid[i]==base_index_bid)
axis=base_data.geom_xpos[base_thumb_gid]-base_data.geom_xpos[base_index_gid]
axis/=np.linalg.norm(axis)
rotation,_=Rotation.align_vectors([axis],[[0,0,1]])
xyzw=rotation.as_quat()
quat=np.r_[xyzw[3],xyzw[:3]]
center=np.array([.04,.16,.78])
root=ET.parse(OUT/screen['scene']).getroot()
ET.SubElement(root.find('worldbody'),'geom',{'name':'bandage_tail_ik_proxy',
    'type':'cylinder','pos':'.04 .16 .78','quat':' '.join(map(str,quat)),
    'size':'.007 .025','rgba':'.95 .89 .70 1','friction':'1 .005 .0001'})
model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
data=mujoco.MjData(model)
thumb_bid=model.body('g1_b/wuji_right/r_thumb_tip_sensor_frame').id
index_bid=model.body('g1_b/wuji_right/r_index_finger_tip_sensor_frame').id
thumb_gid=next(i for i in range(model.ngeom) if model.geom_bodyid[i]==thumb_bid)
index_gid=next(i for i in range(model.ngeom) if model.geom_bodyid[i]==index_bid)
proxy_gid=model.geom('bandage_tail_ik_proxy').id
thumb_goal=center+.020*axis
index_goal=center-.020*axis
names=[n for n in screen['joint_names'] if n.startswith('g1_b/right_')]
names += [model.joint(j).name for j in range(model.njnt) if
          model.joint(j).name.startswith(('g1_b/wuji_right/r_thumb_','g1_b/wuji_right/r_index_')) and
          'abd' not in model.joint(j).name]
ids=[model.joint(n).id for n in names]
qa=np.array([model.jnt_qposadr[j] for j in ids])
lo=model.jnt_range[ids,0]+1e-5
hi=model.jnt_range[ids,1]-1e-5
start=q0[qa].copy()
right_wrist=model.body('g1_b/wuji_right/r_wrist').id
data.qpos[:]=q0;mujoco.mj_forward(model,data)
wrist_start=data.xpos[right_wrist].copy()
rows=[]
for seed_scale in (0.,.3,-.3):
    seed=start.copy()
    seed[:7]+=seed_scale*np.array([.1,.1,.1,.1,.1,.1,.1])
    def residual(x):
        data.qpos[:]=q0
        data.qpos[qa]=x
        mujoco.mj_forward(model,data)
        penetration=0.
        for c in data.contact:
            names_here=[model.geom(g).name for g in c.geom]
            bodies=[model.body(model.geom_bodyid[g]).name for g in c.geom]
            if ('floor' not in names_here and any(n.startswith('g1_b/') for n in bodies) and
                    'bandage_tail_ik_proxy' not in names_here):
                penetration=max(penetration,-float(c.dist))
        thumb_distance=mujoco.mj_geomDistance(model,data,proxy_gid,thumb_gid,1.,None)
        index_distance=mujoco.mj_geomDistance(model,data,proxy_gid,index_gid,1.,None)
        return np.r_[.1*(data.geom_xpos[thumb_gid]-thumb_goal),
                     .1*(data.geom_xpos[index_gid]-index_goal),
                     2*(thumb_distance+.0005),2*(index_distance+.0005),
                     10*max(0.,penetration-.002),
                     .05*(data.xpos[right_wrist]-wrist_start),
                     .002*(x-start)]
    fit=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=250)
    residual(fit.x)
    contacts={'thumb':[],'index':[]};deep=[]
    for c in data.contact:
        geom_names=[model.geom(g).name for g in c.geom]
        body_names=[model.body(model.geom_bodyid[g]).name for g in c.geom]
        if c.dist<-.003 and 'floor' not in geom_names and any(n.startswith('g1_b/') for n in body_names):
            deep.append({'bodies':body_names,'distance_m':float(c.dist)})
        if proxy_gid not in c.geom:continue
        for digit,bid in (('thumb',thumb_bid),('index',index_bid)):
            if model.body(bid).name in body_names:
                contacts[digit].append((float(c.dist),c.frame[:3].copy()*
                                        (1 if c.geom1==proxy_gid else -1)))
    dot=min((float(np.dot(a[1],b[1])) for a in contacts['thumb'] for b in contacts['index']),default=None)
    rows.append({'seed_scale':seed_scale,'optimizer_success':bool(fit.success),
                 'proxy_geom_center_m':data.geom_xpos[proxy_gid].tolist(),
                 'thumb_geom_center_m':data.geom_xpos[thumb_gid].tolist(),
                 'index_geom_center_m':data.geom_xpos[index_gid].tolist(),
                 'thumb_tip_center_error_m':float(np.linalg.norm(data.geom_xpos[thumb_gid]-thumb_goal)),
                 'index_tip_center_error_m':float(np.linalg.norm(data.geom_xpos[index_gid]-index_goal)),
                 'thumb_tip_contact_count':len(contacts['thumb']),
                 'index_tip_contact_count':len(contacts['index']),
                 'minimum_opposing_normal_dot':dot,
                 'deep_nonfloor_contacts':deep,
                 'right_wrist_displacement_m':float(np.linalg.norm(data.xpos[right_wrist]-wrist_start)),
                 'joint_targets_rad':fit.x.tolist()})
report={'scope':'Fixed rigid tail proxy, static arm+digit IK only; no dynamic grasp',
        'proxy_center_m':center.tolist(),'proxy_quaternion_wxyz':quat.tolist(),
        'joint_names':names,'results':rows}
(OUT/'joint_b_tail_arm_digit_ik_screen.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('joint_names','results')}|
    {'results':[{k:v for k,v in row.items() if k not in ('joint_targets_rad','deep_nonfloor_contacts')}|
                 {'deep_contact_count':len(row['deep_nonfloor_contacts'])} for row in rows]}))
