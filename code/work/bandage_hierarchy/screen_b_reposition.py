"""Offline base-placement screen for B's two arm tasks; no gait claim."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_dressing_pinch_post_variant.xml'))
state=np.load(OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz')
q0=state['qpos'].copy()
data=mujoco.MjData(model);data.qpos[:]=q0;mujoco.mj_forward(model,data)
base_qa=model.joint('g1_b/floating_base').qposadr[0]
left=model.body('g1_b/wuji_left/l_wrist').id
right=model.body('g1_b/wuji_right/r_wrist').id
forearm=model.body('training_forearm_dynamic').id
left_goal=data.xpos[left].copy()
left_rotation=data.xmat[left].reshape(3,3).copy()
forearm_pos=data.xpos[forearm].copy()
ids=[j for j in range(model.njnt) if model.joint(j).name.startswith(('g1_b/left_','g1_b/right_'))
     and any(token in model.joint(j).name for token in ('shoulder','elbow','wrist'))]
qa=np.array([model.jnt_qposadr[j] for j in ids])
start=q0[qa].copy();lo=model.jnt_range[ids,0]+1e-5;hi=model.jnt_range[ids,1]-1e-5
forearm_geom=model.geom('manikin_forearm').id
rows=[]
for right_offset in ((.24,0.,-.02),(.24,-.04,-.02),(.28,-.04,-.02),
                     (.28,-.08,-.02),(.28,0.,-.02)):
 for shift_x in (-.10,-.15):
    shifted=q0.copy();shifted[base_qa]+=shift_x
    right_goal=forearm_pos+np.array(right_offset)
    def residual(x):
        data.qpos[:]=shifted;data.qpos[qa]=x;mujoco.mj_forward(model,data)
        current_left_rotation=data.xmat[left].reshape(3,3)
        self_pen=0.;cross_pen=0.;world_pen=0.
        for c in data.contact:
            if c.dist>=-.002:continue
            bodies=[model.body(model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
            if all(body.startswith('g1_b/') for body in bodies):
                self_pen=max(self_pen,-float(c.dist)-.002)
            elif any(body.startswith('g1_a/') for body in bodies) and any(
                    body.startswith('g1_b/') for body in bodies):
                cross_pen=max(cross_pen,-float(c.dist)-.002)
            elif 'world' in bodies and any(body.startswith('g1_b/') for body in bodies) and not any(
                    model.geom(g).name=='floor' for g in (c.geom1,c.geom2)):
                world_pen=max(world_pen,-float(c.dist)-.002)
        return np.r_[data.xpos[left]-left_goal,data.xpos[right]-right_goal,
                     .1*(current_left_rotation-left_rotation).ravel(),
                     10*self_pen,10*cross_pen,10*world_pen,
                     .002*(x-start)]
    candidates=[least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=200)
                for seed in (start,(lo+hi)/2)]
    fit=min(candidates,key=lambda candidate:np.linalg.norm(candidate.fun[:6]))
    residual(fit.x)
    left_error=float(np.linalg.norm(data.xpos[left]-left_goal))
    right_error=float(np.linalg.norm(data.xpos[right]-right_goal))
    cap_center=data.geom_xpos[forearm_geom]
    cap_axis=data.geom_xmat[forearm_geom].reshape(3,3)[:,2]
    cap_u=np.clip(np.dot(data.xpos[right]-cap_center,cap_axis),
                  -model.geom_size[forearm_geom,1],model.geom_size[forearm_geom,1])
    wrist_surface_clearance=float(np.linalg.norm(data.xpos[right]-(cap_center+cap_u*cap_axis))-
                                  model.geom_size[forearm_geom,0])
    current_left_rotation=data.xmat[left].reshape(3,3)
    left_orientation_error=float(np.arccos(np.clip((np.trace(current_left_rotation.T@left_rotation)-1)/2,-1,1)))
    cross_contacts=[float(c.dist) for c in data.contact if c.dist<-.003 and
        any(model.body(model.geom_bodyid[g]).name.startswith('g1_a/') for g in (c.geom1,c.geom2)) and
        any(model.body(model.geom_bodyid[g]).name.startswith('g1_b/') for g in (c.geom1,c.geom2))]
    b_self_contacts=[];b_world_contacts=[]
    for c in data.contact:
        if c.dist>=-.003:continue
        bodies=[model.body(model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
        if all(body.startswith('g1_b/') for body in bodies):
            b_self_contacts.append(float(c.dist))
        elif any(body.startswith('g1_b/') for body in bodies) and 'world' in bodies and not any(
                model.geom(g).name=='floor' for g in (c.geom1,c.geom2)):
            b_world_contacts.append(float(c.dist))
    rows.append({'base_x_shift_m':shift_x,'right_target_offset_from_forearm_m':right_offset,
                 'right_goal_world_m':right_goal.tolist(),
                 'left_wrist_error_m':left_error,
                 'right_wrist_error_m':right_error,
                 'right_wrist_clearance_to_forearm_surface_m':wrist_surface_clearance,
                 'left_wrist_orientation_error_rad':left_orientation_error,
                 'both_wrist_errors_below_20mm':left_error<.02 and right_error<.02,
                 'deep_cross_robot_contact_count':len(cross_contacts),
                 'worst_cross_robot_penetration_m':max((-d for d in cross_contacts),default=0.),
                 'deep_b_self_contact_count':len(b_self_contacts),
                 'worst_b_self_penetration_m':max((-d for d in b_self_contacts),default=0.),
                 'deep_b_world_nonfloor_contact_count':len(b_world_contacts),
                 'collision_screen_pass':not (cross_contacts or b_self_contacts or b_world_contacts),
                 'assist_wrist_screen_pass':(left_error<.02 and right_error<.02 and
                    left_orientation_error<.2 and wrist_surface_clearance<.12 and
                    not (cross_contacts or b_self_contacts or b_world_contacts)),
                 'joint_targets_rad':fit.x.tolist()})
report={'scope':'Static robot placement screen only; floating base and feet are repositioned '
                'without a trajectory, so this is not a feasible walking demonstration',
        'scene':'joint_spool_grasp_dressing_pinch_post_variant.xml',
        'left_goal_world_m':left_goal.tolist(),
        'joint_names':[model.joint(j).name for j in ids], 'results':rows}
name='joint_b_reposition_static_screen.json'
(OUT/name).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps({'output':name,'results':[{k:v for k,v in row.items() if k!='joint_targets_rad'}
                                           for row in rows]}))
