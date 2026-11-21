"""Static planted-feet IK for a collision-screened fourth-hand stance."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--right-offset',type=float,nargs=3,default=(.28,-.04,-.02))
parser.add_argument('--output',default='joint_b_planted_stance_static_audit.json')
parser.add_argument('--right-hand-flex-rad',type=float,default=None,
                    help='Uniform right-hand finger precontact pose; abduction joints set to zero')
parser.add_argument('--base-z-shift-m',type=float,default=0.)
args=parser.parse_args()
screen=json.loads((OUT/'joint_b_reposition_static_screen.json').read_text())
candidate=next(row for row in screen['results'] if row['base_x_shift_m']==-.10 and
               np.allclose(row['right_target_offset_from_forearm_m'],args.right_offset))
model=mujoco.MjModel.from_xml_path(str(OUT/screen['scene']))
state=np.load(OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz')
q0=state['qpos'].copy();data=mujoco.MjData(model);data.qpos[:]=q0;mujoco.mj_forward(model,data)
left_wrist=model.body('g1_b/wuji_left/l_wrist').id
right_wrist=model.body('g1_b/wuji_right/r_wrist').id
left_wrist_goal=data.xpos[left_wrist].copy()
left_wrist_rotation=data.xmat[left_wrist].reshape(3,3).copy()
right_wrist_goal=np.asarray(candidate['right_goal_world_m'])
feet=[model.body(f'g1_b/{side}_ankle_roll_link').id for side in ('left','right')]
foot_positions=[data.xpos[bid].copy() for bid in feet]
foot_rotations=[data.xmat[bid].reshape(3,3).copy() for bid in feet]
base_qa=model.joint('g1_b/floating_base').qposadr[0]
q=q0.copy();q[base_qa]+=-.10;q[base_qa+2]+=args.base_z_shift_m
for name,value in zip(screen['joint_names'],candidate['joint_targets_rad']):
    q[model.joint(name).qposadr[0]]=value
leg_ids=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_b/')
         and any(token in model.joint(j).name for token in ('hip','knee','ankle'))]
qa=np.array([model.jnt_qposadr[j] for j in leg_ids])
lo=model.jnt_range[leg_ids,0]+1e-5;hi=model.jnt_range[leg_ids,1]-1e-5
start=q0[qa].copy()

def residual(x):
    data.qpos[:]=q;data.qpos[qa]=x;mujoco.mj_forward(model,data)
    parts=[]
    for bid,pos,rotation in zip(feet,foot_positions,foot_rotations):
        parts.append(data.xpos[bid]-pos)
        parts.append(.06*(data.xmat[bid].reshape(3,3)-rotation).ravel())
    parts.append(.0001*(x-start))
    return np.concatenate(parts)

fits=[least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=350)
      for seed in (start,(lo+hi)/2)]
fit=min(fits,key=lambda x:np.linalg.norm(x.fun[:24]))
residual(fit.x)
hand_targets={}
if args.right_hand_flex_rad is not None:
    for joint in range(model.njnt):
        name=model.joint(joint).name
        if not name.startswith('g1_b/wuji_right/r_'):continue
        target=0. if 'abd' in name else args.right_hand_flex_rad
        target=float(np.clip(target,model.jnt_range[joint,0]+.01,
                             model.jnt_range[joint,1]-.01))
        data.qpos[model.jnt_qposadr[joint]]=target
        hand_targets[name]=target
    mujoco.mj_forward(model,data)
foot_errors=[float(np.linalg.norm(data.xpos[bid]-pos)) for bid,pos in zip(feet,foot_positions)]
foot_rot_errors=[float(np.arccos(np.clip((np.trace(data.xmat[bid].reshape(3,3).T@rotation)-1)/2,-1,1)))
                 for bid,rotation in zip(feet,foot_rotations)]
left_wrist_error=float(np.linalg.norm(data.xpos[left_wrist]-left_wrist_goal))
right_wrist_error=float(np.linalg.norm(data.xpos[right_wrist]-right_wrist_goal))
left_wrist_orientation_error=float(np.arccos(np.clip((np.trace(
    data.xmat[left_wrist].reshape(3,3).T@left_wrist_rotation)-1)/2,-1,1)))
b_body_ids=[i for i in range(model.nbody) if model.body(i).name.startswith('g1_b/')]
b_mass=np.array([model.body_mass[i] for i in b_body_ids])
b_com=(b_mass[:,None]*data.xipos[b_body_ids]).sum(axis=0)/b_mass.sum()
foot_geom_ids=[i for i in range(model.ngeom) if
               model.geom(i).name.startswith('g1_b/') and
               'official_foot_contact_' in model.geom(i).name]
foot_support_xy=data.geom_xpos[foot_geom_ids,:2].copy()
support_hull=ConvexHull(foot_support_xy)
support_margin=float(min(-(edge[0]*b_com[0]+edge[1]*b_com[1]+edge[2]) /
                         np.linalg.norm(edge[:2]) for edge in support_hull.equations))
deep_contacts=[{'geoms':[model.geom(c.geom1).name,model.geom(c.geom2).name],
                'bodies':[model.body(model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)],
                'distance_m':float(c.dist)} for c in data.contact if c.dist<-.003 and
               any(model.body(model.geom_bodyid[g]).name.startswith('g1_b/') for g in (c.geom1,c.geom2)) and
               'floor' not in (model.geom(c.geom1).name,model.geom(c.geom2).name)]
floor_contact_sides=set()
for contact in data.contact:
    if 'floor' not in (model.geom(contact.geom1).name,model.geom(contact.geom2).name):
        continue
    for geom in (contact.geom1,contact.geom2):
        name=model.body(model.geom_bodyid[geom]).name
        for side in ('left','right'):
            if name.startswith(f'g1_b/{side}_ankle'):
                floor_contact_sides.add(side)
report={'scope':'Static planted-feet IK only; no torque, balance, stepping or closed-loop pass',
        'base_x_shift_m':-.10,'base_z_shift_m':args.base_z_shift_m,
        'source_candidate':candidate,
        'leg_joint_names':[model.joint(j).name for j in leg_ids],
        'leg_joint_targets_rad':fit.x.tolist(),
        'right_hand_precontact_targets_rad':hand_targets,
        'foot_position_errors_m':foot_errors,'foot_rotation_errors_rad':foot_rot_errors,
        'left_wrist_error_m':left_wrist_error,
        'right_wrist_error_m':right_wrist_error,
        'left_wrist_orientation_error_rad':left_wrist_orientation_error,
        'b_center_of_mass_world_m':b_com.tolist(),
        'b_foot_support_points_xy_m':foot_support_xy.tolist(),
        'b_com_support_margin_m':support_margin,
        'static_balance_geometry_pass':support_margin>=.02,
        'foot_centers_world_m':[data.xpos[bid].tolist() for bid in feet],
        'leg_delta_norm_rad':float(np.linalg.norm(fit.x-start)),
        'deep_nonfloor_b_contacts':deep_contacts,
        'floor_contact_sides':sorted(floor_contact_sides),
        'static_foot_pose_pass':max(foot_errors)<.002 and max(foot_rot_errors)<.1 and
                                floor_contact_sides=={'left','right'},
        'static_collision_pass':not deep_contacts,
        'static_placement_pass':max(foot_errors)<.002 and max(foot_rot_errors)<.1 and
                                 floor_contact_sides=={'left','right'} and
                                 not deep_contacts and
                                 support_margin>=.02 and
                                 left_wrist_error<.02 and right_wrist_error<.02 and
                                 left_wrist_orientation_error<.2}
name=args.output
(OUT/name).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k not in
                  ('source_candidate','leg_joint_names','leg_joint_targets_rad',
                   'right_hand_precontact_targets_rad','deep_nonfloor_b_contacts')}
                 |{'deep_nonfloor_b_contact_count':len(deep_contacts)}))
