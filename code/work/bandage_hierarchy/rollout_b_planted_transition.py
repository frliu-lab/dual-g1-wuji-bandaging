"""Closed-loop B posture transition along screened planted-foot path (simulation only)."""
import json
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--duration',type=float,default=4.)
parser.add_argument('--path-file',default='joint_b_planted_transition_synchronous_refined_kinematic_audit.json')
args=parser.parse_args()
sys.path.insert(0,str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC

path=json.loads((OUT/args.path_file).read_text())
screen=json.loads((OUT/'joint_b_reposition_static_screen.json').read_text())
model=mujoco.MjModel.from_xml_path(str(OUT/screen['scene']))
data=mujoco.MjData(model)
points=np.array([r['qpos'] for r in path['samples']])
data.qpos[:]=points[0]
data.qvel[:]=0
mujoco.mj_forward(model,data)
palms=('g1_a/wuji_left/l_wrist','g1_a/wuji_right/r_wrist',
       'g1_b/wuji_left/l_wrist','g1_b/wuji_right/r_wrist')
def robot_contact_only(m,c):
    return any(m.body(m.geom_bodyid[int(g)]).name.startswith(('g1_a/','g1_b/'))
               for g in c.geom if g>=0)
wbc=JointWBC(model,data.qpos.copy(),palm_names=palms,contact_filter=robot_contact_only)
right_id=model.body('g1_b/wuji_right/r_wrist').id
left_id=model.body('g1_b/wuji_left/l_wrist').id
right_goal=np.asarray(path['samples'][-1]['qpos'])
ref_data=mujoco.MjData(model);ref_data.qpos[:]=right_goal;mujoco.mj_forward(model,ref_data)
right_target=ref_data.xpos[right_id].copy()
left_target=ref_data.xpos[left_id].copy()
root_id=model.body('g1_b/pelvis').id
foot_ids=[i for i in range(model.ngeom) if model.geom(i).name.startswith('g1_b/')
          and 'official_foot_contact_' in model.geom(i).name]
initial_feet=data.geom_xpos[foot_ids].copy()
min_joint_margin=float('inf');min_joint_name=None;min_joint_time=None
max_nonfloor_pen=0.;max_root_speed=0.;max_foot_drift=0.
support_frames=0;control_solves=0;error=None
role_contact_frames={role:0 for role in ('a_left_forearm','a_right_dressing','b_left_spool','b_right_forearm')}
duration=args.duration;dt=.02;substeps=round(dt/model.opt.timestep)
for k in range(round(duration/dt)):
    u=np.clip((k*dt)/(duration*.75),0,1)
    smooth=u*u*u*(10+u*(-15+6*u))
    s=smooth*(len(points)-1)
    i=min(int(s),len(points)-2);f=s-i
    wbc.qref[:]=(1-f)*points[i]+f*points[i+1]
    try:
        ctrl,meta,_=wbc.solve(data)
        data.ctrl[:]=ctrl
        control_solves+=1
        for _ in range(substeps):
            mujoco.mj_step(model,data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                raise RuntimeError('Nonfinite state')
            max_root_speed=max(max_root_speed,float(np.linalg.norm(data.cvel[root_id,3:])))
            max_foot_drift=max(max_foot_drift,float(np.max(np.linalg.norm(
                data.geom_xpos[foot_ids,:2]-initial_feet[:,:2],axis=1))))
            floor_sides=set()
            roles_this_step=set()
            for c in data.contact:
                names=[model.geom(g).name for g in c.geom]
                bodies=[model.body(model.geom_bodyid[g]).name for g in c.geom]
                if 'floor' in names:
                    for side in ('left','right'):
                        if any(n.startswith(f'g1_b/{side}_official_foot_contact_') for n in names):
                            floor_sides.add(side)
                elif any(n.startswith('g1_b/') for n in bodies):
                    max_nonfloor_pen=max(max_nonfloor_pen,-float(c.dist))
                if c.dist<=.001:
                    if 'training_forearm_dynamic' in bodies and any(n.startswith('g1_a/wuji_left/') for n in bodies):
                        roles_this_step.add('a_left_forearm')
                    if any(n.startswith('g1_a/wuji_right/') for n in bodies) and any(
                            n in ('dressing_pad','dressing_pad_dynamic') for n in bodies):
                        roles_this_step.add('a_right_dressing')
                    if any(n.startswith('g1_b/wuji_left/') for n in bodies) and any(
                            n.startswith(('bandage_', 'shared_spool')) for n in bodies):
                        roles_this_step.add('b_left_spool')
                    if 'training_forearm_dynamic' in bodies and any(n.startswith('g1_b/wuji_right/') for n in bodies):
                        roles_this_step.add('b_right_forearm')
            for role in roles_this_step:
                role_contact_frames[role]+=1
            support_frames+=floor_sides=={'left','right'}
            for j in range(model.njnt):
                if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[j] and model.joint(j).name.startswith('g1_b/'):
                    q=data.qpos[model.jnt_qposadr[j]]
                    margin=min(float(q-model.jnt_range[j,0]),float(model.jnt_range[j,1]-q))
                    if margin<min_joint_margin:
                        min_joint_margin=margin
                        min_joint_name=model.joint(j).name
                        min_joint_time=float(data.time)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        break
report={'scope':'Simulated planted-foot B posture transition from saved checkpoint; no footsteps or cloth wrapping',
        'source_path':args.path_file,
        'planned_duration_s':duration,'simulated_duration_s':float(data.time),
        'control_solves':control_solves,'error':error,
        'b_two_foot_support_fraction':support_frames/max(1,control_solves*substeps),
        'b_max_nonfloor_penetration_m':max_nonfloor_pen,
        'b_max_foot_marker_xy_drift_m':max_foot_drift,
        'b_max_pelvis_speed_m_s':max_root_speed,
        'b_min_joint_limit_margin_rad':min_joint_margin,
        'b_min_joint_limit_name':min_joint_name,'b_min_joint_limit_time_s':min_joint_time,
        'b_final_right_wrist_error_m':float(np.linalg.norm(data.xpos[right_id]-right_target)),
        'b_final_left_wrist_error_m':float(np.linalg.norm(data.xpos[left_id]-left_target))}
report['role_contact_fractions']={k:v/max(1,control_solves*substeps) for k,v in role_contact_frames.items()}
suffix='_lowered10mm' if 'lowered10mm' in args.path_file else ''
(OUT/f'joint_b_planted_transition_wbc_{duration:g}s{suffix}_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
