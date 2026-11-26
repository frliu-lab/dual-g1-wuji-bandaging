"""Test whether the earlier A-right dressing hold pose transfers to the flex scene."""
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'))
d=mujoco.MjData(m)
current=np.load(OUT/'joint_b_pinchtab_rrt_path.npz')['qpos'][-1]
validated=np.load(OUT/'joint_handle_grasp_first_joint_pass_qpos.npz')['qpos'][-1]
d.qpos[:167]=current[:167]
indices=[]
for j in range(m.njnt):
    name=m.joint(j).name
    if not name.startswith(('g1_a/right_','g1_a/wuji_right/')):continue
    adr=int(m.jnt_qposadr[j])
    if adr<167:
        indices.append(adr)
        d.qpos[adr]=validated[adr]
mujoco.mj_forward(m,d)
contacts={'thumb':[],'index':[]}
max_pen={'a':0.,'b':0.}
for c in d.contact:
    geoms=[m.geom(int(g)).name for g in c.geom if g>=0]
    bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
    if 'floor' not in geoms:
        for agent in ('a','b'):
            if any(name.startswith(f'g1_{agent}/') for name in bodies):
                max_pen[agent]=max(max_pen[agent],max(0.,-float(c.dist)))
    if 'dressing_pad' in bodies:
        for digit in contacts:
            if any(name.startswith(f'g1_a/wuji_right/r_{digit}_') for name in bodies):
                contacts[digit].append({'body':next(name for name in bodies if name.startswith('g1_a/')),
                                        'distance_m':float(c.dist),
                                        'normal':c.frame[:3].tolist()})
report={'scope':'Static transfer of previously validated A-right arm/finger pose; no dynamic execution',
        'joint_count':len(indices),'max_joint_change_rad':float(np.max(abs(d.qpos[indices]-current[indices]))),
        'contacts':contacts,'max_nonfloor_penetration_m':max_pen}
candidate_states=np.load(OUT/'joint_dressing_index_tip_candidates.npz')['qpos']
index_indices=[int(m.jnt_qposadr[j]) for j in range(m.njnt)
               if m.joint(j).name.startswith('g1_a/wuji_right/r_index_finger_')]
candidate_audits=[]
for candidate_id,state in enumerate(candidate_states):
    d.qpos[:167]=current[:167]
    d.qpos[index_indices]=state[index_indices]
    mujoco.mj_forward(m,d)
    pad_contacts={'thumb':[],'index':[]};normals={'thumb':np.zeros(3),'index':np.zeros(3)};candidate_pen=0.
    for ci,c in enumerate(d.contact):
        geoms=[m.geom(int(g)).name for g in c.geom if g>=0]
        bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
        if 'floor' not in geoms and any(name.startswith(('g1_a/','g1_b/')) for name in bodies):
            candidate_pen=max(candidate_pen,max(0.,-float(c.dist)))
        if 'dressing_pad' in bodies:
            for digit in pad_contacts:
                if any(name.startswith(f'g1_a/wuji_right/r_{digit}_') for name in bodies):
                    pad_contacts[digit].append(float(c.dist))
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                    sign=1. if c.geom1>=0 and m.geom(c.geom1).name=='dressing_pad_contact' else -1.
                    normals[digit]+=sign*max(0.,float(wrench[0]))*c.frame[:3]
    denominator=float(np.linalg.norm(normals['thumb'])*np.linalg.norm(normals['index']))
    cosine=float(normals['thumb']@normals['index']/denominator) if denominator>1e-10 else None
    candidate_audits.append({'candidate_id':candidate_id,'max_joint_change_rad':float(np.max(abs(d.qpos[index_indices]-current[index_indices]))),
                             'thumb_contact_count':len(pad_contacts['thumb']),'index_contact_count':len(pad_contacts['index']),
                             'contact_normal_cosine':cosine,
                             'max_nonfloor_penetration_m':candidate_pen})
report['index_candidate_audits']=candidate_audits
side=np.load(OUT/'pad_side_pinch_candidates.npz')
old_model=mujoco.MjModel.from_xml_path(str(OUT/'pad_side_pinch_variant.xml'))
old_addresses=set(map(int,side['joint_qpos_addresses']))
side_audits=[]
for candidate_id,state in enumerate(side['qpos']):
    d.qpos[:167]=current[:167]
    changed=[]
    for j in range(old_model.njnt):
        old_addr=int(old_model.jnt_qposadr[j]);name=old_model.joint(j).name
        if old_addr not in old_addresses or not name.startswith(('g1_a/right_','g1_a/wuji_right/')):continue
        new_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
        if new_id<0:continue
        new_addr=int(m.jnt_qposadr[new_id]);d.qpos[new_addr]=state[old_addr];changed.append(new_addr)
    mujoco.mj_forward(m,d)
    normals={'thumb':np.zeros(3),'index':np.zeros(3)};counts={'thumb':0,'index':0};deep=0.
    for ci,c in enumerate(d.contact):
        geoms=[m.geom(int(g)).name for g in c.geom if g>=0]
        bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
        if 'floor' not in geoms and any(name.startswith(('g1_a/','g1_b/')) for name in bodies):deep=max(deep,max(0.,-float(c.dist)))
        if 'dressing_pad' not in bodies:continue
        for digit in normals:
            if any(name.startswith(f'g1_a/wuji_right/r_{digit}_') for name in bodies):
                wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                sign=1. if c.geom1>=0 and m.geom(c.geom1).name=='dressing_pad_contact' else -1.
                normals[digit]+=sign*max(0.,float(wrench[0]))*c.frame[:3];counts[digit]+=1
    denominator=float(np.linalg.norm(normals['thumb'])*np.linalg.norm(normals['index']))
    side_audits.append({'candidate_id':candidate_id,'transferred_joint_count':len(changed),
                        'max_joint_change_rad':float(np.max(abs(d.qpos[changed]-current[changed]))),
                        'contacts':counts,'normal_cosine':float(normals['thumb']@normals['index']/denominator) if denominator>1e-10 else None,
                        'max_nonfloor_penetration_m':deep})
report['side_pinch_candidate_audits']=side_audits
(OUT/'joint_b_flex_a_right_pose_transfer_static_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'contact_count':{k:len(v) for k,v in contacts.items()},'index_candidate_audits':candidate_audits,
                  'side_pinch_candidate_audits':side_audits}))
