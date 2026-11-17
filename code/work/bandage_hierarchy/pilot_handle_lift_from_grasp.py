"""Restart from a validated simulated grasp to tune a short pickup controller.

This is an isolated checkpoint pilot. It does not count as a full-sequence pass.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--lift-m',type=float,default=.01)
parser.add_argument('--duration-s',type=float,default=2.)
parser.add_argument('--kp-n-per-m',type=float,default=20.)
parser.add_argument('--kd-n-s-per-m',type=float,default=2.)
parser.add_argument('--stick',action='store_true')
parser.add_argument('--lateral-feedback',action='store_true')
args=parser.parse_args()
if not 0.<args.lift_m<=.03 or not 1.<=args.duration_s<=3.:parser.error('Invalid lift profile')
sys.path.insert(0,str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC

model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
data=mujoco.MjData(model)
state=np.load(OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz')
data.qpos[:]=state['qpos'];data.qvel[:]=state['qvel'];mujoco.mj_forward(model,data)
base=data.qpos.copy()
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
wbc.b_handle_stick=args.stick
arm_joints=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_b/left_') and
            any(s in model.joint(j).name for s in ('shoulder','elbow','wrist'))]
arm_qa=np.array([model.jnt_qposadr[j] for j in arm_joints])
arm_va=np.array([model.jnt_dofadr[j] for j in arm_joints])
wrist=model.body('g1_b/wuji_left/l_wrist').id
spool=model.body('bandage_spool').id
spool_vz=model.joint('bandage_spool_free').dofadr[0]+2
spool_vxy=model.joint('bandage_spool_free').dofadr[0]
start_z=float(data.xpos[spool,2]);mass=float(model.body_subtreemass[spool]);weight=mass*abs(model.opt.gravity[2])
start_xy=data.xpos[spool,:2].copy()
ref=mujoco.MjData(model)
digit_info={}
spool_rot=data.xmat[spool].reshape(3,3)
for digit,label in (('thumb','thumb'),('index','index_finger')):
    body=model.body(f'g1_b/wuji_left/l_{label}_tip_sensor_frame').id
    joints=[j for j in range(model.njnt) if model.joint(j).name.startswith(f'g1_b/wuji_left/l_{label}_')]
    digit_info[digit]={'body':body,'qa':np.array([model.jnt_qposadr[j] for j in joints]),
                       'va':np.array([model.jnt_dofadr[j] for j in joints]),
                       'local':spool_rot.T@(data.xpos[body]-data.xpos[spool])}
interval=round(.02/model.opt.timestep)
rows=[];errors=[]
for step in range(round(args.duration_s*50)):
    t=step*.02
    desired_z=start_z+args.lift_m*min(1.,t/args.duration_s)
    desired_v=args.lift_m/args.duration_s
    actual_z=float(data.xpos[spool,2]);actual_v=float(data.qvel[spool_vz])
    desired_force=float(np.clip(weight+args.kp_n_per_m*(desired_z-actual_z)+
                                args.kd_n_s_per_m*(desired_v-actual_v),0.,1.5))
    wbc.b_handle_lift_min_n=max(0.,desired_force-.05)
    wbc.b_handle_lift_force_goal_n=desired_force
    if args.lateral_feedback:
        wbc.b_handle_lateral_force_target_n=np.clip(-40.*(data.xpos[spool,:2]-start_xy)-
                                                     2.*data.qvel[spool_vxy:spool_vxy+2],-1.,1.)
    wbc.qref[:]=base
    ref.qpos[:]=base;mujoco.mj_forward(model,ref)
    jp=np.zeros((3,model.nv));mujoco.mj_jac(model,ref,jp,None,ref.xpos[wrist],wrist)
    J=jp[:,arm_va]
    correction=J.T@np.linalg.solve(J@J.T+1e-4*np.eye(3),np.array([0.,0.,desired_z-start_z]))
    wbc.qref[arm_qa]+=np.clip(correction,-.12,.12)
    rot=data.xmat[spool].reshape(3,3)
    for info in digit_info.values():
        body=info['body'];Jpos=np.zeros((3,model.nv))
        mujoco.mj_jac(model,data,Jpos,None,data.xpos[body],body)
        jac=Jpos[:,info['va']]
        target=data.xpos[spool]+rot@info['local']
        delta=np.linalg.solve(jac.T@jac+1e-3*np.eye(jac.shape[1]),jac.T@(.6*(target-data.xpos[body])))
        wbc.qref[info['qa']]+=np.clip(delta,-.08,.08)
    try:
        data.ctrl[:],_,_=wbc.solve(data)
        for _ in range(interval):mujoco.mj_step(model,data)
    except RuntimeError as exc:
        errors.append(str(exc));break
    if not np.all(np.isfinite(data.qpos)):
        errors.append('nonfinite_state');break
    rows.append({'t_s':float(data.time),'desired_z_m':desired_z,
                 'spool_z_m':float(data.xpos[spool,2]),'spool_vz_m_s':float(data.qvel[spool_vz]),
                 'spool_xy_m':data.xpos[spool,:2].tolist(),
                 'lateral_shift_m':float(np.linalg.norm(data.xpos[spool,:2]-start_xy)),
                 'desired_up_force_n':desired_force,
                 'wrist_z_m':float(data.xpos[wrist,2])})
    if rows[-1]['lateral_shift_m']>.005:
        errors.append('spool_lateral_displacement_over_5mm');break
    if abs(float(data.xpos[spool,2])-desired_z)>.008:
        errors.append('height_tracking_error_over_8mm');break
report={'scope':'Checkpoint restart from validated simulated grasp; not full-sequence bandaging',
        'lift_m':args.lift_m,'duration_s':args.duration_s,'kp_n_per_m':args.kp_n_per_m,
        'kd_n_s_per_m':args.kd_n_s_per_m,'stick':args.stick,'spool_mass_kg':mass,
        'lateral_feedback':args.lateral_feedback,
        'start_spool_z_m':start_z,'final_spool_z_m':rows[-1]['spool_z_m'] if rows else start_z,
        'max_spool_z_m':max((r['spool_z_m'] for r in rows),default=start_z),
        'max_lateral_shift_m':max((r['lateral_shift_m'] for r in rows),default=0.),
        'completed':len(rows)==round(args.duration_s*50) and not errors,
        'errors':errors,'trace':rows}
stem=f'joint_handle_lift_checkpoint_{"stick" if args.stick else "free"}_kp{args.kp_n_per_m:g}_kd{args.kd_n_s_per_m:g}'+('_lateral' if args.lateral_feedback else '')
(OUT/f'{stem}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
