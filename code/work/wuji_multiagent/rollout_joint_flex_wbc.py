"""Short closed-loop dual-G1 WBC rollout with a prewrapped flexible strip."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np

from joint_wbc import JointWBC

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--radius-mm',type=float,default=48.)
parser.add_argument('--phase-deg',type=float,default=135.)
parser.add_argument('--flex-contact-weight',type=float,default=10.)
parser.add_argument('--thickness-mm',type=float,default=2.)
parser.add_argument('--duration-s',type=float,default=.2)
args=parser.parse_args()
variant=f'_radius{args.radius_mm:g}mm_phase{args.phase_deg:g}deg'
if args.thickness_mm!=2.:
    variant+=f'_thickness{args.thickness_mm:g}mm'
model=mujoco.MjModel.from_xml_path(str(OUT/f'joint_dual_g1_prewrapped_flex_scene_world_anchored{variant}.xml'))
data=mujoco.MjData(model)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
q=np.asarray(seed['samples'][-1]['qpos'])
data.qpos[:len(q)]=q
mujoco.mj_forward(model,data)
palms=('g1_a/wuji_left/l_wrist','g1_a/wuji_right/r_wrist',
       'g1_b/wuji_left/l_wrist','g1_b/wuji_right/r_wrist')
def relevant_contact(m,c):
    return any(f>=0 for f in c.flex) or any(
        m.body(m.geom_bodyid[g]).name.startswith(('g1_a/','g1_b/'))
        for g in c.geom if g>=0)
wbc=JointWBC(model,data.qpos.copy(),palm_names=palms,contact_filter=relevant_contact)
wbc.backend='DIRECT'
wbc.flex_contact_acceleration_weight=args.flex_contact_weight
initial_roots={name:data.xpos[model.body(name).id].copy() for name in ('g1_a/pelvis','g1_b/pelvis')}
initial_free_edge=data.flexvert_xpos[64:66].mean(axis=0).copy()
duration=args.duration_s;interval=round(.02/model.opt.timestep)
error=None;solves=0;max_pen=0.;max_robot_flex_pen=0.;max_flex_forearm_pen=0.
feet_steps=0;flex_forearm_steps=0;max_contact_residual=0.;max_free_edge_drift=0.;trace=[]
for _ in range(round(duration/.02)):
    try:
        ctrl,meta,_=wbc.solve(data)
        data.ctrl[:]=ctrl
        solves+=1
        max_contact_residual=max(max_contact_residual,meta['contact_acceleration_task_residual_m_s2'])
        interval_flex_forearm=0
        for _ in range(interval):
            mujoco.mj_step(model,data)
            if not np.all(np.isfinite(data.qpos)):
                raise RuntimeError('Nonfinite state')
            max_free_edge_drift=max(max_free_edge_drift,float(np.linalg.norm(
                data.flexvert_xpos[64:66].mean(axis=0)-initial_free_edge)))
            feet={'a':set(),'b':set()}
            has_flex_forearm=False
            for c in data.contact:
                rigid=next((g for g in c.geom if g>=0),None)
                body=model.body(model.geom_bodyid[rigid]).name if rigid is not None else ''
                flex=any(f>=0 for f in c.flex)
                if flex and body=='training_forearm_dynamic':
                    has_flex_forearm=True
                    max_flex_forearm_pen=max(max_flex_forearm_pen,-float(c.dist))
                if flex and body.startswith(('g1_a/','g1_b/')):
                    max_robot_flex_pen=max(max_robot_flex_pen,-float(c.dist))
                if c.dist<0 and 'floor' not in [model.geom(g).name for g in c.geom if g>=0]:
                    max_pen=max(max_pen,-float(c.dist))
                names=[model.geom(g).name for g in c.geom if g>=0]
                if 'floor' in names:
                    for agent in ('a','b'):
                        for side in ('left','right'):
                            if any(n.startswith(f'g1_{agent}/{side}_official_foot_contact_') for n in names):
                                feet[agent].add(side)
            feet_steps+=all(feet[agent]=={'left','right'} for agent in ('a','b'))
            flex_forearm_steps+=has_flex_forearm
            interval_flex_forearm+=has_flex_forearm
        trace.append({'time_s':float(data.time),'flex_forearm_contact_fraction':interval_flex_forearm/interval,
                      'contact_task_residual_m_s2':meta['contact_acceleration_task_residual_m_s2']})
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        break
steps=max(1,solves*interval)
forearm_id=model.body('training_forearm_dynamic').id
origin=data.xpos[forearm_id]
rotation=data.xmat[forearm_id].reshape(3,3)
local=(data.flexvert_xpos[:66].reshape(33,2,3).mean(axis=1)-origin)@rotation
axis=np.array([.34,-.17,-.04]);axis/=np.linalg.norm(axis)
radial_a=np.cross(axis,[0.,0.,1.]);radial_a/=np.linalg.norm(radial_a)
radial_b=np.cross(axis,radial_a)
offset=local-np.array([.17,-.085,-.02])
radial=offset-np.outer(offset@axis,axis)
radius=np.linalg.norm(radial,axis=1)
near=(radius>=.046)&(radius<=.054)
angles=np.mod(np.arctan2(radial@radial_b,radial@radial_a),2*np.pi)
bins=np.floor(12*angles/(2*np.pi)).astype(int).clip(0,11)
coverage=[len(set(bins[lo:hi][near[lo:hi]]))/12 for lo,hi in ((0,17),(16,33))]
report={'scope':'Dual-G1 WBC hold of a prewrapped deformable strip; no robot-generated wrap',
        'radius_mm':args.radius_mm,'phase_deg':args.phase_deg,
        'strip_collision_thickness_mm':args.thickness_mm,
        'flex_contact_weight':args.flex_contact_weight,
        'duration_s':float(data.time),'control_solves':solves,'error':error,
        'two_robot_two_foot_fraction':feet_steps/steps,
        'flex_forearm_contact_fraction':flex_forearm_steps/steps,
        'max_nonfloor_penetration_m':max_pen,
        'max_robot_flex_penetration_m':max_robot_flex_pen,
        'max_flex_forearm_penetration_m':max_flex_forearm_pen,
        'max_free_edge_drift_m':max_free_edge_drift,
        'final_near_forearm_centerline_fraction':float(near.mean()),
        'final_angular_coverage_per_turn_12_bins':coverage,
        'max_contact_acceleration_task_residual_m_s2':max_contact_residual,
        'root_drift_m':{name:float(np.linalg.norm(data.xpos[model.body(name).id]-start))
                        for name,start in initial_roots.items()},
        'trace':trace,'finite_state':bool(np.all(np.isfinite(data.qpos)))}
(OUT/f'joint_flex_wbc_{args.duration_s:g}s_rollout{variant}{"_flexweight"+str(args.flex_contact_weight).replace(".","p") if args.flex_contact_weight!=10. else ""}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
