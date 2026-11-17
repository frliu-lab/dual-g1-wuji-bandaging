"""Diagnostic A-right dressing pickup from a verified double-G1 grasp checkpoint.

Only a short checkpoint pilot. Full-sequence replay and contact validation are required.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
sys.path.insert(0,str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC

p=argparse.ArgumentParser()
p.add_argument('--lift-m',type=float,default=.01)
p.add_argument('--duration-s',type=float,default=1.)
p.add_argument('--dressing-force-goal-n',type=float,default=.7)
p.add_argument('--index-closure-m',type=float,default=0.)
p.add_argument('--index-pregrasp-candidate',type=int,default=-1)
p.add_argument('--index-preposition-s',type=float,default=.5)
p.add_argument('--index-weight',type=float,default=1.)
p.add_argument('--dressing-lift-min-n',type=float,default=0.)
args=p.parse_args()
if not 0.<args.lift_m<=.03 or not .5<=args.duration_s<=3. or args.index_preposition_s>=args.duration_s:
    p.error('Lift range (0,.03] m; duration [.5,3] s; preposition shorter than duration')
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
data=mujoco.MjData(model)
state=np.load(OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz')
data.qpos[:]=state['qpos'];data.qvel[:]=state['qvel'];mujoco.mj_forward(model,data)
base=state['reference_qpos'].copy()
palms=('g1_a/wuji_left/l_wrist','g1_a/wuji_right/r_wrist','g1_b/wuji_left/l_wrist','g1_b/wuji_right/r_wrist')
def contact_filter(m,c):
    return any(m.body(m.geom_bodyid[int(g)]).name.startswith(('g1_a/','g1_b/')) for g in c.geom if g>=0)
wbc=JointWBC(model,base,palm_names=palms,contact_filter=contact_filter,
             palm_rotation_weights=[10.,10.,10.,10.],
             palm_position_weights=[1000.,1500.,1000.,1000.])
wbc.forearm_contact_force_limit_n=12.
wbc.forearm_finger_force_limit_n=2.
wbc.support_palm_force_goal_n=11.5
wbc.b_spool_contact_force_cap_n=1.5
wbc.dressing_contact_normal_goal_n=args.dressing_force_goal_n
arm_joints=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_a/right_') and
            any(s in model.joint(j).name for s in ('shoulder','elbow','wrist'))]
arm_qa=np.array([model.jnt_qposadr[j] for j in arm_joints])
arm_va=np.array([model.jnt_dofadr[j] for j in arm_joints])
wrist=model.body('g1_a/wuji_right/r_wrist').id
pad=model.body('dressing_pad').id
spool=model.body('bandage_spool').id
start_pad=data.xpos[pad].copy();start_spool=data.xpos[spool].copy()
ref=mujoco.MjData(model)
digit_info={}
pad_rot=data.xmat[pad].reshape(3,3)
for digit,label in (('thumb','thumb'),('index','index_finger')):
    body=model.body(f'g1_a/wuji_right/r_{label}_tip_sensor_frame').id
    joints=[j for j in range(model.njnt) if model.joint(j).name.startswith(f'g1_a/wuji_right/r_{label}_')]
    digit_info[digit]={'body':body,'qa':np.array([model.jnt_qposadr[j] for j in joints]),
                       'va':np.array([model.jnt_dofadr[j] for j in joints]),
                       'local':pad_rot.T@(data.xpos[body]-start_pad)}
index_candidate=(np.load(OUT/'joint_dressing_index_tip_candidates.npz')['qpos'][args.index_pregrasp_candidate]
                 if args.index_pregrasp_candidate>=0 else None)
wbc.weights[digit_info['index']['va']]=args.index_weight
index_local_captured=False
interval=round(.02/model.opt.timestep)
rows=[];errors=[]
for step in range(round(args.duration_s*50)):
    t=step*.02
    lift_elapsed=max(0.,t-(args.index_preposition_s if index_candidate is not None else 0.))
    lift_duration=args.duration_s-(args.index_preposition_s if index_candidate is not None else 0.)
    planned=args.lift_m*min(1.,lift_elapsed/lift_duration)
    wbc.dressing_lift_min_n=(args.dressing_lift_min_n*min(1.,lift_elapsed/.2)
                              if args.dressing_lift_min_n and lift_elapsed>0. else None)
    wbc.qref[:]=base
    if index_candidate is not None:
        qa=digit_info['index']['qa']
        blend=min(1.,t/args.index_preposition_s)
        wbc.qref[qa]=(1-blend)*base[qa]+blend*index_candidate[qa]
    ref.qpos[:]=base;mujoco.mj_forward(model,ref)
    jp=np.zeros((3,model.nv));mujoco.mj_jac(model,ref,jp,None,ref.xpos[wrist],wrist)
    J=jp[:,arm_va]
    correction=J.T@np.linalg.solve(J@J.T+1e-4*np.eye(3),np.array([0.,0.,planned]))
    wbc.qref[arm_qa]+=np.clip(correction,-.12,.12)
    rot=data.xmat[pad].reshape(3,3)
    for info in digit_info.values():
        if info is digit_info['index'] and index_candidate is not None:
            if t<args.index_preposition_s:continue
            if not index_local_captured:
                info['local']=rot.T@(data.xpos[info['body']]-data.xpos[pad])
                index_local_captured=True
        body=info['body'];jp=np.zeros((3,model.nv))
        mujoco.mj_jac(model,data,jp,None,data.xpos[body],body)
        jac=jp[:,info['va']]
        local=info['local'].copy()
        if info is digit_info['index'] and args.index_closure_m:
            local-=args.index_closure_m*local/max(1e-6,np.linalg.norm(local))
        target=data.xpos[pad]+rot@local
        delta=np.linalg.solve(jac.T@jac+1e-3*np.eye(jac.shape[1]),jac.T@(.6*(target-data.xpos[body])))
        wbc.qref[info['qa']]+=np.clip(delta,-.08,.08)
    try:
        data.ctrl[:],_,_=wbc.solve(data)
        for _ in range(interval):mujoco.mj_step(model,data)
    except RuntimeError as exc:
        errors.append(str(exc));break
    rows.append({'time_s':float(data.time),'pad_center_m':data.xpos[pad].tolist(),
                 'pad_lift_m':float(data.xpos[pad,2]-start_pad[2]),
                 'pad_lateral_shift_m':float(np.linalg.norm(data.xpos[pad,:2]-start_pad[:2])),
                 'spool_shift_m':float(np.linalg.norm(data.xpos[spool]-start_spool))})
    rows[-1]['digit_tip_m']={name:data.xpos[info['body']].tolist() for name,info in digit_info.items()}
    rows[-1]['digit_joint_error_rad']={name:(data.qpos[info['qa']]-wbc.qref[info['qa']]).tolist()
                                         for name,info in digit_info.items()}
    rows[-1]['pad_contacts']=[{
        'geoms':[model.geom(c.geom1).name,model.geom(c.geom2).name],
        'bodies':[model.body(model.geom_bodyid[c.geom1]).name,
                  model.body(model.geom_bodyid[c.geom2]).name],
        'distance_m':float(c.dist)} for c in data.contact
        if 'dressing_pad_contact' in (model.geom(c.geom1).name,model.geom(c.geom2).name)]
    if rows[-1]['pad_lateral_shift_m']>.01:errors.append('pad_lateral_shift_over_10mm');break
    if rows[-1]['spool_shift_m']>.005:errors.append('spool_shift_over_5mm');break
report={'scope':'Checkpoint diagnostic; no full-sequence dressing pickup claim',
        'lift_m':args.lift_m,'duration_s':args.duration_s,
        'dressing_force_goal_n':args.dressing_force_goal_n,
        'index_closure_m':args.index_closure_m,
        'index_pregrasp_candidate':args.index_pregrasp_candidate,
        'index_weight':args.index_weight,
        'dressing_lift_min_n':args.dressing_lift_min_n,
        'start_pad_center_m':start_pad.tolist(),
        'final_pad_center_m':rows[-1]['pad_center_m'] if rows else start_pad.tolist(),
        'final_pad_lift_m':rows[-1]['pad_lift_m'] if rows else 0.,
        'max_pad_lateral_shift_m':max((r['pad_lateral_shift_m'] for r in rows),default=0.),
        'completed':len(rows)==round(args.duration_s*50) and not errors,'errors':errors,'trace':rows}
report['final_pad_contacts']=rows[-1]['pad_contacts'] if rows else []
stem=f'joint_dressing_pickup_checkpoint_lift{args.lift_m:g}_force{args.dressing_force_goal_n:g}_iclose{args.index_closure_m:g}_ipre{args.index_pregrasp_candidate}_iw{args.index_weight:g}_up{args.dressing_lift_min_n:g}'
(OUT/f'{stem}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
np.savez_compressed(OUT/f'{stem}_final_state.npz',qpos=data.qpos,qvel=data.qvel)
print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
