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
from support_quality import SupportQualityMonitor, SupportSample

p=argparse.ArgumentParser()
p.add_argument('--lift-m',type=float,default=.01)
p.add_argument('--duration-s',type=float,default=1.)
p.add_argument('--motion-end-s',type=float,default=None,
               help='End arm translation before the pilot ends, then hold the lifted pose')
p.add_argument('--dressing-force-goal-n',type=float,default=.7)
p.add_argument('--index-closure-m',type=float,default=0.)
p.add_argument('--index-pregrasp-candidate',type=int,default=-1)
p.add_argument('--side-pregrasp-candidate',type=int,default=-1)
p.add_argument('--index-preposition-s',type=float,default=.5)
p.add_argument('--index-weight',type=float,default=1.)
p.add_argument('--dressing-lift-min-n',type=float,default=0.)
p.add_argument('--dressing-force-start-s',type=float,default=0.)
p.add_argument('--dressing-force-end-s',type=float,default=None)
p.add_argument('--track-pad-pose',action='store_true')
p.add_argument('--track-pad-orientation',action='store_true')
p.add_argument('--pad-pose-weight',type=float,default=10.)
p.add_argument('--wrist-x-shift-m',type=float,default=0.)
p.add_argument('--support-wrist-yaw-offset-rad',type=float,default=0.)
p.add_argument('--support-wrist-yaw-start-s',type=float,default=0.)
p.add_argument('--start-state-token',default='75fdd3a3b4b1',
               choices=('75fdd3a3b4b1','947858479e10'),
               help='15.7 s joint grasp or 17.7 s full-sequence 20 mm spool pickup endpoint')
p.add_argument('--b-hold-min-force-n',type=float,default=.35)
p.add_argument('--b-hold-force-goal-n',type=float,default=.45)
args=p.parse_args()
if not 0.<args.lift_m<=.06 or not .5<=args.duration_s<=3. or args.index_preposition_s>=args.duration_s:
    p.error('Lift range (0,.06] m; duration [.5,3] s; preposition shorter than duration')
motion_end=args.motion_end_s if args.motion_end_s is not None else args.duration_s
if not args.index_preposition_s<motion_end<=args.duration_s:p.error('motion end must follow preposition and precede duration')
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
data=mujoco.MjData(model)
state=np.load(OUT/f'joint_support_pinch_staged_wbc_report_spool_settled_handle_{args.start_state_token}_final_state.npz')
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
if args.start_state_token=='947858479e10':
    wbc.b_handle_stick=True
    wbc.b_handle_lift_min_n=args.b_hold_min_force_n or None
    wbc.b_handle_lift_force_goal_n=args.b_hold_force_goal_n or None
arm_joints=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_a/right_') and
            any(s in model.joint(j).name for s in ('shoulder','elbow','wrist'))]
arm_qa=np.array([model.jnt_qposadr[j] for j in arm_joints])
arm_va=np.array([model.jnt_dofadr[j] for j in arm_joints])
b_hold_qa=np.array([model.jnt_qposadr[j] for j in range(model.njnt)
                    if model.joint(j).name.startswith('g1_b/left_') and
                    any(s in model.joint(j).name for s in ('shoulder','elbow','wrist'))],dtype=int)
b_hand_hold_qa=np.array([model.jnt_qposadr[j] for j in range(model.njnt)
                         if model.joint(j).name.startswith('g1_b/wuji_left/l_')],dtype=int)
wrist=model.body('g1_a/wuji_right/r_wrist').id
pad=model.body('dressing_pad').id
pad_free=model.joint('dressing_pad_free')
pad_qadr=pad_free.qposadr[0];pad_vadr=pad_free.dofadr[0]
if args.track_pad_pose:wbc.weights[pad_vadr:pad_vadr+6]=args.pad_pose_weight
elif args.track_pad_orientation:wbc.weights[pad_vadr+3:pad_vadr+6]=args.pad_pose_weight
spool=model.body('bandage_spool').id
start_pad=data.xpos[pad].copy();start_spool=data.xpos[spool].copy()
spool_free=model.joint('bandage_spool_free')
spool_mass=float(model.body_subtreemass[spool])
ref=mujoco.MjData(model)
digit_info={}
spool_digit_info={}
if args.start_state_token=='947858479e10':
    spool_rot=data.xmat[spool].reshape(3,3)
    for digit,label in (('thumb','thumb'),('index','index_finger')):
        body=model.body(f'g1_b/wuji_left/l_{label}_tip_sensor_frame').id
        joints=[j for j in range(model.njnt) if model.joint(j).name.startswith(f'g1_b/wuji_left/l_{label}_')]
        spool_digit_info[digit]={'body':body,'qa':np.array([model.jnt_qposadr[j] for j in joints]),
                                 'va':np.array([model.jnt_dofadr[j] for j in joints]),
                                 'local':spool_rot.T@(data.xpos[body]-start_spool)}
pad_rot=data.xmat[pad].reshape(3,3)
for digit,label in (('thumb','thumb'),('index','index_finger')):
    body=model.body(f'g1_a/wuji_right/r_{label}_tip_sensor_frame').id
    joints=[j for j in range(model.njnt) if model.joint(j).name.startswith(f'g1_a/wuji_right/r_{label}_')]
    digit_info[digit]={'body':body,'qa':np.array([model.jnt_qposadr[j] for j in joints]),
                       'va':np.array([model.jnt_dofadr[j] for j in joints]),
                       'local':pad_rot.T@(data.xpos[body]-start_pad)}
index_candidate=(np.load(OUT/'joint_dressing_index_tip_candidates.npz')['qpos'][args.index_pregrasp_candidate]
                 if args.index_pregrasp_candidate>=0 else None)
side_candidate=(np.load(OUT/'joint_dressing_side_grasp_from_joint_candidates_refined.npz')['qpos'][args.side_pregrasp_candidate]
                if args.side_pregrasp_candidate>=0 else None)
pregrasp_candidate=side_candidate if side_candidate is not None else index_candidate
wbc.weights[digit_info['index']['va']]=args.index_weight
local_captured={name:False for name in digit_info}
interval=round(.02/model.opt.timestep)
def contact_force(ci):
    wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
    return max(0.,float(wrench[0]))
rows=[];errors=[]
recorded_qpos=[]
support_monitor=SupportQualityMonitor()
support_yaw_qa=model.joint('g1_a/left_wrist_yaw_joint').qposadr[0]
elbow=model.joint('training_elbow_hinge')
for step in range(round(args.duration_s*50)):
    t=step*.02
    lift_elapsed=max(0.,t-(args.index_preposition_s if pregrasp_candidate is not None else 0.))
    lift_duration=motion_end-(args.index_preposition_s if pregrasp_candidate is not None else 0.)
    planned=args.lift_m*min(1.,lift_elapsed/lift_duration)
    force_elapsed=max(0.,t-args.dressing_force_start_s)
    wbc.dressing_lift_min_n=(args.dressing_lift_min_n*min(1.,force_elapsed/.2)
                              if args.dressing_lift_min_n and force_elapsed>0. and
                              (args.dressing_force_end_s is None or t<args.dressing_force_end_s) else None)
    if args.start_state_token=='947858479e10':
        spool_velocity=float(data.qvel[spool_free.dofadr[0]+2])
        spool_error=float(start_spool[2]-data.xpos[spool,2])
        hold_force=float(np.clip(spool_mass*abs(model.opt.gravity[2])+60.*spool_error-2.*spool_velocity,0.,1.5))
        wbc.b_handle_lift_min_n=(max(0.,hold_force-.05) if args.b_hold_min_force_n else None)
        wbc.b_handle_lift_force_goal_n=hold_force if args.b_hold_force_goal_n else None
        wbc.b_handle_lateral_force_target_n=np.clip(-40.*(data.xpos[spool,:2]-start_spool[:2])-
            2.*data.qvel[spool_free.dofadr[0]:spool_free.dofadr[0]+2],-1.,1.)
    wbc.qref[:]=base
    if args.start_state_token=='947858479e10':
        wbc.qref[b_hold_qa]=state['qpos'][b_hold_qa]
        wbc.qref[b_hand_hold_qa]=state['qpos'][b_hand_hold_qa]
    wbc.qref[support_yaw_qa]+=args.support_wrist_yaw_offset_rad*min(1.,max(0.,t-args.support_wrist_yaw_start_s)/.5)
    wbc.qref[elbow.qposadr[0]]=.25
    if args.track_pad_pose:wbc.qref[pad_qadr+2]=base[pad_qadr+2]+planned
    if pregrasp_candidate is not None:
        blend=min(1.,t/args.index_preposition_s)
        for name,info in digit_info.items():
            if name=='index' or side_candidate is not None:
                qa=info['qa']
                wbc.qref[qa]=(1-blend)*base[qa]+blend*pregrasp_candidate[qa]
    ref.qpos[:]=base;mujoco.mj_forward(model,ref)
    jp=np.zeros((3,model.nv));mujoco.mj_jac(model,ref,jp,None,ref.xpos[wrist],wrist)
    J=jp[:,arm_va]
    correction=J.T@np.linalg.solve(J@J.T+1e-4*np.eye(3),
                                   np.array([args.wrist_x_shift_m*min(1.,lift_elapsed/lift_duration),0.,planned]))
    wbc.qref[arm_qa]+=np.clip(correction,-.12,.12)
    for info in spool_digit_info.values():
        body=info['body'];rot_spool=data.xmat[spool].reshape(3,3)
        target=data.xpos[spool]+rot_spool@info['local']
        jp=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jp,None,data.xpos[body],body)
        jac=jp[:,info['va']]
        delta=np.linalg.solve(jac.T@jac+1e-3*np.eye(len(info['va'])),jac.T@(.6*(target-data.xpos[body])))
        wbc.qref[info['qa']]+=np.clip(delta,-.08,.08)
    rot=data.xmat[pad].reshape(3,3)
    for name,info in digit_info.items():
        if pregrasp_candidate is not None and (name=='index' or side_candidate is not None):
            if t<args.index_preposition_s:continue
            if not local_captured[name]:
                info['local']=rot.T@(data.xpos[info['body']]-data.xpos[pad])
                local_captured[name]=True
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
        impulses={};feet=set();deep=False;support_peak=0.;spool_grip={}
        for _ in range(interval):
            mujoco.mj_step(model,data)
            frame_support=0.
            for ci,c in enumerate(data.contact):
                if c.geom1<0 or c.geom2<0:continue
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                        model.body(model.geom_bodyid[c.geom2]).name)
                if 'floor' in geoms:
                    for robot in ('g1_a','g1_b'):
                        if any(b.startswith(robot+'/') for b in bodies):feet.add(robot)
                elif any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
                    deep=True
                if 'manikin_forearm' in geoms:
                    hand=next((b for b in bodies if b.startswith('g1_a/wuji_left/')),None)
                    if hand:
                        normal=contact_force(ci)
                        impulses[hand]=impulses.get(hand,0.)+normal*model.opt.timestep
                        frame_support+=normal
                if any('bandage_spool' in g or 'bandage_handle_prototype' in g for g in geoms):
                    hand=next((b for b in bodies if b.startswith('g1_b/wuji_left/')),None)
                    if hand:spool_grip[hand]=spool_grip.get(hand,0.)+contact_force(ci)*model.opt.timestep
            support_peak=max(support_peak,frame_support)
        support_valid=support_monitor.update(SupportSample(float(data.time),
            float(data.qpos[elbow.qposadr[0]]),float(data.qvel[elbow.dofadr[0]]),
            support_peak,impulses,feet=={'g1_a','g1_b'},deep))
    except RuntimeError as exc:
        errors.append(str(exc));break
    rows.append({'time_s':float(data.time),'pad_center_m':data.xpos[pad].tolist(),
                 'pad_lift_m':float(data.xpos[pad,2]-start_pad[2]),
                 'pad_lateral_shift_m':float(np.linalg.norm(data.xpos[pad,:2]-start_pad[:2])),
                 'spool_shift_m':float(np.linalg.norm(data.xpos[spool]-start_spool))})
    recorded_qpos.append(data.qpos.copy())
    rows[-1].update({'support_valid':bool(support_valid),'support_quality':support_monitor.last_report,
                     'feet_supported':feet=={'g1_a','g1_b'},'deep_robot_contact':deep,
                     'spool_grip_impulse_by_body_ns':spool_grip})
    rows[-1]['digit_tip_m']={name:data.xpos[info['body']].tolist() for name,info in digit_info.items()}
    rows[-1]['digit_joint_error_rad']={name:(data.qpos[info['qa']]-wbc.qref[info['qa']]).tolist()
                                         for name,info in digit_info.items()}
    rows[-1]['pad_contacts']=[{
        'geoms':[model.geom(c.geom1).name,model.geom(c.geom2).name],
        'bodies':[model.body(model.geom_bodyid[c.geom1]).name,
                  model.body(model.geom_bodyid[c.geom2]).name],
        'distance_m':float(c.dist),
        'normal_force_n':contact_force(ci),
        'pad_to_other_normal_world':((1. if model.geom(c.geom1).name=='dressing_pad_contact' else -1.)*
                                      np.asarray(c.frame[:3])).tolist()} for ci,c in enumerate(data.contact)
        if 'dressing_pad_contact' in (model.geom(c.geom1).name,model.geom(c.geom2).name)]
    if rows[-1]['pad_lateral_shift_m']>max(.01,abs(args.wrist_x_shift_m)+.01):
        errors.append('pad_lateral_shift_over_plan_plus_10mm');break
    if rows[-1]['spool_shift_m']>.005:errors.append('spool_shift_over_5mm');break
report={'scope':'Checkpoint diagnostic; no full-sequence dressing pickup claim',
        'lift_m':args.lift_m,'duration_s':args.duration_s,
        'motion_end_s':motion_end,
        'dressing_force_goal_n':args.dressing_force_goal_n,
        'index_closure_m':args.index_closure_m,
        'index_pregrasp_candidate':args.index_pregrasp_candidate,
        'side_pregrasp_candidate':args.side_pregrasp_candidate,
        'index_weight':args.index_weight,
        'dressing_lift_min_n':args.dressing_lift_min_n,
        'dressing_force_start_s':args.dressing_force_start_s,
        'dressing_force_end_s':args.dressing_force_end_s,
        'track_pad_pose':args.track_pad_pose,
        'track_pad_orientation':args.track_pad_orientation,
        'pad_pose_weight':args.pad_pose_weight,
        'wrist_x_shift_m':args.wrist_x_shift_m,
        'support_wrist_yaw_offset_rad':args.support_wrist_yaw_offset_rad,
        'support_wrist_yaw_start_s':args.support_wrist_yaw_start_s,
        'start_state_token':args.start_state_token,
        'b_hold_min_force_n':args.b_hold_min_force_n,
        'b_hold_force_goal_n':args.b_hold_force_goal_n,
        'final_support_quality':support_monitor.last_report,
        'all_feet_supported':all(r['feet_supported'] for r in rows),
        'any_deep_robot_contact':any(r['deep_robot_contact'] for r in rows),
        'start_pad_center_m':start_pad.tolist(),
        'final_pad_center_m':rows[-1]['pad_center_m'] if rows else start_pad.tolist(),
        'final_pad_lift_m':rows[-1]['pad_lift_m'] if rows else 0.,
        'max_pad_lateral_shift_m':max((r['pad_lateral_shift_m'] for r in rows),default=0.),
        'completed':len(rows)==round(args.duration_s*50) and not errors,'errors':errors,'trace':rows}
report['final_pad_contacts']=rows[-1]['pad_contacts'] if rows else []
stem=f'joint_dressing_pickup_checkpoint_lift{args.lift_m:g}_force{args.dressing_force_goal_n:g}_iclose{args.index_closure_m:g}_ipre{args.index_pregrasp_candidate}_side{args.side_pregrasp_candidate}_iw{args.index_weight:g}_up{args.dressing_lift_min_n:g}_start{args.dressing_force_start_s:g}'
if args.motion_end_s is not None:stem+=f'_motionend{args.motion_end_s:g}'
if args.dressing_force_end_s is not None:stem+=f'_forceend{args.dressing_force_end_s:g}'
if args.track_pad_pose:stem+=f'_pose{args.pad_pose_weight:g}'
if args.track_pad_orientation:stem+=f'_orient{args.pad_pose_weight:g}'
if args.wrist_x_shift_m:stem+=f'_xshift{args.wrist_x_shift_m:g}'
if args.support_wrist_yaw_offset_rad:stem+=f'_syaw{args.support_wrist_yaw_offset_rad:g}_at{args.support_wrist_yaw_start_s:g}'
stem+=f'_from{args.start_state_token}'
if args.start_state_token=='947858479e10':stem+=f'_bmin{args.b_hold_min_force_n:g}_bgoal{args.b_hold_force_goal_n:g}'
(OUT/f'{stem}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
np.savez_compressed(OUT/f'{stem}_final_state.npz',qpos=data.qpos,qvel=data.qvel)
np.savez_compressed(OUT/f'{stem}_qpos.npz',qpos=np.asarray(recorded_qpos),
                    time_s=np.asarray([row['time_s'] for row in rows]))
print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
