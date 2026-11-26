"""Check prior B-left spool-grasp arm pose inside the current flex scene."""
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'))
current=np.load(OUT/'joint_b_pinchtab_rrt_path.npz')['qpos'][-1]
validated=np.load(OUT/'joint_handle_grasp_first_joint_pass_qpos.npz')['qpos'][-1]
a_seed=np.load(OUT/'joint_b_flex_a_thumb_side_contact_candidates.npz')['qpos'][0]
cases=[]
for transfer in (False,True):
    d=mujoco.MjData(m);d.qpos[:167]=current[:167]
    for j in range(m.njnt):
        name=m.joint(j).name;adr=int(m.jnt_qposadr[j])
        if name.startswith(('g1_a/wuji_right/r_thumb_','g1_a/wuji_right/r_index_finger_')):d.qpos[adr]=a_seed[adr]
        if transfer and name.startswith('g1_b/left_') and any(t in name for t in ('shoulder','elbow','wrist')):d.qpos[adr]=validated[adr]
    mujoco.mj_forward(m,d)
    normals={'thumb':[],'index':[]};max_pen=0.
    for c in d.contact:
        geoms=[m.geom(int(g)).name for g in c.geom if g>=0]
        bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
        if 'floor' not in geoms and any(name.startswith(('g1_a/','g1_b/')) for name in bodies):max_pen=max(max_pen,max(0.,-float(c.dist)))
        if 'bandage_spool' in bodies:
            for digit in normals:
                if any(name.startswith(f'g1_b/wuji_left/l_{digit}_') for name in bodies):
                    sign=1. if c.geom1>=0 and m.geom(c.geom1).name=='bandage_handle_prototype' else -1.
                    normals[digit].append(sign*c.frame[:3].copy())
    cases.append({'transfer_b_left_arm':transfer,'contacts':{k:len(v) for k,v in normals.items()},
                  'opposed_pair_count':sum(float(a@b)<-.5 for a in normals['thumb'] for b in normals['index']),
                  'max_nonfloor_penetration_m':max_pen})
report={'scope':'Static transfer only; B-left approach and dynamic spool grasp unverified','cases':cases}
(OUT/'joint_b_flex_b_left_handle_pose_transfer_static_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
