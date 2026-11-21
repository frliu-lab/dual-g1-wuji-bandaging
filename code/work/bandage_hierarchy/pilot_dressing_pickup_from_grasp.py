"""Diagnostic A-right dressing pickup from a verified double-G1 grasp checkpoint.

Only a short checkpoint pilot. Full-sequence replay and contact validation are required.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
sys.path.insert(0,str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC
from support_quality import SupportQualityMonitor, SupportSample
from pickup_phase_gate import evaluate_pickup_gate

p=argparse.ArgumentParser()
p.add_argument('--lift-m',type=float,default=.01)
p.add_argument('--duration-s',type=float,default=1.)
p.add_argument('--motion-end-s',type=float,default=None,
               help='End arm translation before the pilot ends, then hold the lifted pose')
p.add_argument('--smooth-motion',action='store_true',help='Cubic ease in/out for the pickup translation')
p.add_argument('--dressing-force-goal-n',type=float,default=.7)
p.add_argument('--dressing-thumb-goal-n',type=float,default=None)
p.add_argument('--dressing-index-goal-n',type=float,default=None)
p.add_argument('--index-closure-m',type=float,default=0.)
p.add_argument('--index-closure-after-s',type=float,default=0.)
p.add_argument('--index-tip-offset-m',type=float,nargs=3,default=(0.,0.,0.),
               help='World-frame index tip target correction during regrasp')
p.add_argument('--index-tip-offset-start-s',type=float,default=.5)
p.add_argument('--thumb-tip-offset-m',type=float,nargs=3,default=(0.,0.,0.),
               help='World-frame thumb tip target correction toward the pad side')
p.add_argument('--thumb-tip-offset-start-s',type=float,default=.5)
p.add_argument('--thumb-closure-m',type=float,default=0.)
p.add_argument('--thumb-closure-after-s',type=float,default=0.)
p.add_argument('--freeze-digit-frame-after-s',type=float,default=None)
p.add_argument('--index-pregrasp-candidate',type=int,default=-1)
p.add_argument('--side-pregrasp-candidate',type=int,default=-1)
p.add_argument('--middle-support-candidate',type=int,default=-1)
p.add_argument('--middle-after-s',type=float,default=None,
               help='Move middle finger only after the initial thumb/index pickup')
p.add_argument('--index-preposition-s',type=float,default=.5)
p.add_argument('--index-weight',type=float,default=1.)
p.add_argument('--pickup-wrist-position-weight',type=float,default=1500.)
p.add_argument('--pickup-wrist-rotation-weight',type=float,default=10.)
p.add_argument('--pickup-wrist-pitch-offset-rad',type=float,default=0.)
p.add_argument('--pickup-wrist-yaw-offset-rad',type=float,default=0.)
p.add_argument('--pickup-wrist-rotate-start-s',type=float,default=.8)
p.add_argument('--pickup-arm-offsets-rad',type=float,nargs=7,default=(0.,)*7,
               help='Right shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw corrections')
p.add_argument('--index-ik-target-rad',type=float,nargs=4,default=None,
               help='Static grasp IK target for index MCP flex/abd, PIP, DIP')
p.add_argument('--thumb-ik-target-rad',type=float,nargs=4,default=None,
               help='Static side-underpad IK target for thumb CMC flex/abd, MCP, IP')
p.add_argument('--transport-grasp-after-s',type=float,default=None,
               help='After grasp, move digit targets with planned pickup instead of observed pad')
p.add_argument('--transport-extra-lift-m',type=float,default=0.)
p.add_argument('--unload-start-s',type=float,default=None,
               help='Start a small finger-and-wrist upward unload before gated transport')
p.add_argument('--unload-lift-m',type=float,default=0.)
p.add_argument('--unload-ramp-s',type=float,default=.6)
p.add_argument('--contact-gated-transport',action='store_true',
               help='Begin transport only after five consecutive measured grip/unload frames')
p.add_argument('--late-tab-pinch-after-s',type=float,default=None)
p.add_argument('--late-thumb-ik-target-rad',type=float,nargs=4,default=None)
p.add_argument('--late-index-ik-target-rad',type=float,nargs=4,default=None)
p.add_argument('--late-arm-offsets-rad',type=float,nargs=7,default=(0.,)*7)
p.add_argument('--online-tab-ik-after-s',type=float,default=None)
p.add_argument('--online-tab-target-z-m',type=float,default=-.006)
p.add_argument('--online-tab-side-offset-m',type=float,default=.008)
p.add_argument('--online-tab-thumb-offset-m',type=float,default=None)
p.add_argument('--online-tab-index-offset-m',type=float,default=None)
p.add_argument('--online-tab-middle-clearance-m',type=float,default=None,
               help='Keep the index middle link radially clear of the diagnostic post')
p.add_argument('--online-tab-ik-transport',action='store_true',
               help='Keep online side-pinch IK during planned upward transport')
p.add_argument('--online-tab-middle',action='store_true',help='Bring middle finger to post with online IK')
p.add_argument('--online-tab-wrist-hold-weight',type=float,default=.2)
p.add_argument('--online-tab-up-force-feedback-gain',type=float,default=0.,
               help='Causal fingertip target-height correction in m/N from measured pad upward force')
p.add_argument('--online-tab-up-force-target-n',type=float,default=.15)
p.add_argument('--online-tab-index-dip-min-rad',type=float,default=None,
               help='Optional natural-flexion lower bound for the online index DIP IK')
p.add_argument('--online-tab-tip-task-weight',type=float,default=0.,
               help='WBC Cartesian fingertip target weight; zero disables the task')
p.add_argument('--tip-first-candidate-index',type=int,default=None,
               help='Replay a collision-screened static post IK target after online IK start')
p.add_argument('--dressing-lift-min-n',type=float,default=0.)
p.add_argument('--dressing-up-goal-n',type=float,default=None)
p.add_argument('--dressing-up-goal-start-s',type=float,default=0.)
p.add_argument('--dressing-force-start-s',type=float,default=0.)
p.add_argument('--dressing-force-end-s',type=float,default=None)
p.add_argument('--track-pad-pose',action='store_true')
p.add_argument('--track-pad-orientation',action='store_true')
p.add_argument('--pad-pose-weight',type=float,default=10.)
p.add_argument('--pad-orientation-after-s',type=float,default=None)
p.add_argument('--pad-orientation-use-start',action='store_true',
               help='Hold checkpoint pad orientation instead of the orientation at activation')
p.add_argument('--wrist-x-shift-m',type=float,default=0.)
p.add_argument('--max-pad-lateral-shift-m',type=float,default=None,
               help='Diagnostic stop threshold; reports still retain measured lateral drift')
p.add_argument('--pad-lateral-feedback-gain',type=float,default=0.,
               help='Soft contact-force correction in N/m toward starting pad XY')
p.add_argument('--pad-lateral-feedback-start-s',type=float,default=0.)
p.add_argument('--support-wrist-yaw-offset-rad',type=float,default=0.)
p.add_argument('--support-wrist-yaw-start-s',type=float,default=0.)
p.add_argument('--support-wrist-yaw-late-offset-rad',type=float,default=0.)
p.add_argument('--support-wrist-yaw-late-start-s',type=float,default=2.5)
p.add_argument('--support-follow-gain',type=float,default=0.)
p.add_argument('--support-follow-start-s',type=float,default=0.)
p.add_argument('--support-pinky-flex-late-offset-rad',type=float,default=0.)
p.add_argument('--support-pinky-late-start-s',type=float,default=2.5)
p.add_argument('--support-palm-shift-late-m',type=float,nargs=3,default=(0.,0.,0.))
p.add_argument('--support-palm-shift-start-s',type=float,default=2.5)
p.add_argument('--support-palm-shift-second-m',type=float,nargs=3,default=(0.,0.,0.))
p.add_argument('--support-palm-shift-second-start-s',type=float,default=3.)
p.add_argument('--support-palm-shift-third-m',type=float,nargs=3,default=(0.,0.,0.))
p.add_argument('--support-palm-shift-third-start-s',type=float,default=3.5)
p.add_argument('--support-palm-position-weight-late',type=float,default=None)
p.add_argument('--support-palm-position-weight-late-start-s',type=float,default=3.5)
p.add_argument('--elbow-hold-weight',type=float,default=1.)
p.add_argument('--support-force-goal-late-n',type=float,default=None)
p.add_argument('--support-force-goal-late-start-s',type=float,default=3.5)
p.add_argument('--finger-forearm-cap-late-n',type=float,default=None)
p.add_argument('--finger-forearm-cap-late-start-s',type=float,default=3.5)
p.add_argument('--tag',default='',help='Distinct suffix for experimental output files')
p.add_argument('--pull-tab-scene',action='store_true',help='Use isolated physical dressing pull-tab scene')
p.add_argument('--pinch-post-scene',action='store_true',help='Use isolated cylindrical pinch-post diagnostic scene')
p.add_argument('--start-state-token',default='75fdd3a3b4b1',
               choices=('75fdd3a3b4b1','947858479e10'),
               help='15.7 s joint grasp or 17.7 s full-sequence 20 mm spool pickup endpoint')
p.add_argument('--b-hold-min-force-n',type=float,default=.35)
p.add_argument('--b-hold-force-goal-n',type=float,default=.45)
args=p.parse_args()
if not 0.<args.lift_m<=.06 or not .5<=args.duration_s<=5. or args.index_preposition_s>=args.duration_s:
    p.error('Lift range (0,.06] m; duration [.5,5] s; preposition shorter than duration')
motion_end=args.motion_end_s if args.motion_end_s is not None else args.duration_s
if not args.index_preposition_s<motion_end<=args.duration_s:p.error('motion end must follow preposition and precede duration')
if args.pull_tab_scene and args.pinch_post_scene:p.error('Choose one diagnostic scene')
if args.unload_lift_m<0. or args.unload_lift_m>.015 or args.unload_ramp_s<=0.:
    p.error('unload lift must be 0-15 mm and ramp positive')
if not 0. <= args.online_tab_up_force_feedback_gain <= .1:
    p.error('online force feedback gain must be in [0, 0.1] m/N')
if args.online_tab_tip_task_weight<0.:p.error('tip task weight must be nonnegative')
scene_name=('joint_spool_grasp_dressing_pinch_post_variant.xml' if args.pinch_post_scene else
            'joint_spool_grasp_dressing_pull_tab_variant.xml' if args.pull_tab_scene else
            'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml')
model=mujoco.MjModel.from_xml_path(str(OUT/scene_name))
data=mujoco.MjData(model)
state=np.load(OUT/f'joint_support_pinch_staged_wbc_report_spool_settled_handle_{args.start_state_token}_final_state.npz')
data.qpos[:]=state['qpos'];data.qvel[:]=state['qvel'];mujoco.mj_forward(model,data)
base=state['reference_qpos'].copy()
palms=('g1_a/wuji_left/l_wrist','g1_a/wuji_right/r_wrist','g1_b/wuji_left/l_wrist','g1_b/wuji_right/r_wrist')
def contact_filter(m,c):
    return any(m.body(m.geom_bodyid[int(g)]).name.startswith(('g1_a/','g1_b/')) for g in c.geom if g>=0)
wbc=JointWBC(model,base,palm_names=palms,contact_filter=contact_filter,
             palm_rotation_weights=[10.,args.pickup_wrist_rotation_weight,10.,10.],
             palm_position_weights=[1000.,args.pickup_wrist_position_weight,1000.,1000.])
wbc.forearm_contact_force_limit_n=12.
wbc.forearm_finger_force_limit_n=2.
wbc.support_palm_force_goal_n=11.5
wbc.b_spool_contact_force_cap_n=1.5
wbc.dressing_contact_normal_goal_n=args.dressing_force_goal_n
if args.dressing_thumb_goal_n is not None or args.dressing_index_goal_n is not None:
    wbc.dressing_digit_normal_goals_n={k:v for k,v in (('thumb',args.dressing_thumb_goal_n),
        ('index',args.dressing_index_goal_n)) if v is not None}
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
original_pad_orientation_weights=wbc.weights[pad_vadr+3:pad_vadr+6].copy()
spool=model.body('bandage_spool').id
start_pad=data.xpos[pad].copy();start_spool=data.xpos[spool].copy()
start_pad_quat=data.qpos[pad_qadr+3:pad_qadr+7].copy()
spool_free=model.joint('bandage_spool_free')
spool_mass=float(model.body_subtreemass[spool])
ref=mujoco.MjData(model)
support_ref=mujoco.MjData(model);support_ref.qpos[:]=base;mujoco.mj_forward(model,support_ref)
forearm_body=model.body('training_forearm_dynamic').id
left_palm=model.body('g1_a/wuji_left/l_wrist').id
left_joints=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_a/left_')
             and any(s in model.joint(j).name for s in ('shoulder','elbow','wrist'))]
left_qa=np.array([model.jnt_qposadr[j] for j in left_joints])
left_va=np.array([model.jnt_dofadr[j] for j in left_joints])
goal_forearm_pos=support_ref.xpos[forearm_body].copy()
goal_forearm_rot=support_ref.xmat[forearm_body].reshape(3,3).copy()
goal_palm_pos=support_ref.xpos[left_palm].copy()
relative_palm_pos=goal_forearm_rot.T@(goal_palm_pos-goal_forearm_pos)
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
for digit,label in ((('thumb','thumb'),('index','index_finger'),('middle','middle_finger'))
                    if args.middle_support_candidate>=0 else (('thumb','thumb'),('index','index_finger'))):
    body=model.body(f'g1_a/wuji_right/r_{label}_tip_sensor_frame').id
    joints=[j for j in range(model.njnt) if model.joint(j).name.startswith(f'g1_a/wuji_right/r_{label}_')]
    digit_info[digit]={'body':body,'qa':np.array([model.jnt_qposadr[j] for j in joints]),
                       'va':np.array([model.jnt_dofadr[j] for j in joints]),
                       'local':pad_rot.T@(data.xpos[body]-start_pad)}
index_candidate=(np.load(OUT/'joint_dressing_index_tip_candidates.npz')['qpos'][args.index_pregrasp_candidate]
                 if args.index_pregrasp_candidate>=0 else None)
side_candidate=(np.load(OUT/'joint_dressing_side_grasp_from_joint_candidates_refined.npz')['qpos'][args.side_pregrasp_candidate]
                if args.side_pregrasp_candidate>=0 else None)
middle_candidate=(np.load(OUT/'joint_dressing_middle_support_candidates.npz')['qpos'][args.middle_support_candidate]
                  if args.middle_support_candidate>=0 else None)
pregrasp_candidate=(side_candidate if args.middle_after_s is not None and side_candidate is not None else
                    middle_candidate if middle_candidate is not None else
                    side_candidate if side_candidate is not None else index_candidate)
wbc.weights[digit_info['index']['va']]=args.index_weight
local_captured={name:False for name in digit_info}
if args.online_tab_ik_after_s is not None:
    if not (args.pull_tab_scene or args.pinch_post_scene):p.error('online tab IK requires diagnostic grip scene')
    ik_data=mujoco.MjData(model)
    ik_ids=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_a/right_')
            and any(s in model.joint(j).name for s in ('shoulder','elbow','wrist'))]
    for prefix in (('r_thumb_','r_index_finger_','r_middle_finger_')
                   if args.online_tab_middle else ('r_thumb_','r_index_finger_')):
        ik_ids += [j for j in range(model.njnt) if model.joint(j).name.startswith('g1_a/wuji_right/'+prefix)]
    if args.online_tab_middle and 'middle' not in digit_info:p.error('online middle requires --middle-support-candidate')
    if args.tip_first_candidate_index is not None and args.online_tab_middle:
        p.error('tip-first candidate currently contains arm, thumb and index only')
    ik_qa=np.array([model.jnt_qposadr[j] for j in ik_ids]);ik_lo=model.jnt_range[ik_ids,0]+1e-5
    ik_hi=model.jnt_range[ik_ids,1]-1e-5
    if args.online_tab_index_dip_min_rad is not None:
        dip_name='g1_a/wuji_right/r_index_finger_dip'
        dip_i=next(i for i,j in enumerate(ik_ids) if model.joint(j).name==dip_name)
        ik_lo[dip_i]=max(ik_lo[dip_i],args.online_tab_index_dip_min_rad)
        if ik_lo[dip_i]>=ik_hi[dip_i]:p.error('index DIP minimum exceeds joint range')
    tab_geom=model.geom('dressing_pull_tab').id
    tip_first_target=None
    if args.tip_first_candidate_index is not None:
        search=json.loads((OUT/'joint_post_tip_first_ik_search.json').read_text(encoding='utf8'))
        selected=search['results'][args.tip_first_candidate_index]
        if search['joint_names']!=[model.joint(j).name for j in ik_ids]:
            p.error('tip-first search joint order differs from pilot IK')
        if selected['max_post_penetration_m']>.001:
            p.error('tip-first candidate has more than 1 mm static post penetration')
        tip_first_target=np.asarray(selected['qpos'])
interval=round(.02/model.opt.timestep)
def contact_force(ci):
    wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
    return max(0.,float(wrench[0]))
rows=[];errors=[]
recorded_qpos=[]
pad_hold_quat=None
fixed_digit_rot=None
transport_anchor=None
transport_ready_streak=0
transport_gate_last=None
tab_feedback_z=0.
effective_tab_target_z=args.online_tab_target_z_m
support_monitor=SupportQualityMonitor()
support_yaw_qa=model.joint('g1_a/left_wrist_yaw_joint').qposadr[0]
support_pinky_qa=model.joint('g1_a/wuji_left/l_pinky_mcp_flex').qposadr[0]
elbow=model.joint('training_elbow_hinge')
wbc.weights[elbow.dofadr[0]]=args.elbow_hold_weight
for step in range(round(args.duration_s*50)):
    t=step*.02
    online_ik_diagnostic=None
    wbc.tip_position_targets={}
    lift_elapsed=max(0.,t-(args.index_preposition_s if pregrasp_candidate is not None else 0.))
    lift_duration=motion_end-(args.index_preposition_s if pregrasp_candidate is not None else 0.)
    motion_u=min(1.,lift_elapsed/lift_duration)
    if args.smooth_motion:motion_u=motion_u*motion_u*(3.-2.*motion_u)
    planned=args.lift_m*motion_u
    unload_offset=(args.unload_lift_m*np.clip((t-args.unload_start_s)/args.unload_ramp_s,0.,1.)
                   if args.unload_start_s is not None else 0.)
    planned+=unload_offset
    if (args.transport_grasp_after_s is not None and t>=args.transport_grasp_after_s and
            (not args.contact_gated_transport or transport_anchor is not None)):
        planned+=args.transport_extra_lift_m*np.clip((t-args.transport_grasp_after_s)/.8,0.,1.)
    force_elapsed=max(0.,t-args.dressing_force_start_s)
    wbc.dressing_lift_min_n=(args.dressing_lift_min_n*min(1.,force_elapsed/.2)
                              if args.dressing_lift_min_n and force_elapsed>0. and
                              (args.dressing_force_end_s is None or t<args.dressing_force_end_s) else None)
    wbc.dressing_up_force_goal_n=(args.dressing_up_goal_n
        if args.dressing_up_goal_n is not None and t>=args.dressing_up_goal_start_s else None)
    wbc.dressing_lateral_force_target_n=(
        np.clip(-args.pad_lateral_feedback_gain*(data.xpos[pad,:2]-start_pad[:2])-
                2.*data.qvel[pad_vadr:pad_vadr+2],-1.,1.)
        if args.pad_lateral_feedback_gain>0. and t>=args.pad_lateral_feedback_start_s else None)
    if args.start_state_token=='947858479e10':
        spool_velocity=float(data.qvel[spool_free.dofadr[0]+2])
        spool_error=float(start_spool[2]-data.xpos[spool,2])
        hold_force=float(np.clip(spool_mass*abs(model.opt.gravity[2])+60.*spool_error-2.*spool_velocity,0.,1.5))
        wbc.b_handle_lift_min_n=(max(0.,hold_force-.05) if args.b_hold_min_force_n else None)
        wbc.b_handle_lift_force_goal_n=hold_force if args.b_hold_force_goal_n else None
        wbc.b_handle_lateral_force_target_n=np.clip(-40.*(data.xpos[spool,:2]-start_spool[:2])-
            2.*data.qvel[spool_free.dofadr[0]:spool_free.dofadr[0]+2],-1.,1.)
    wbc.qref[:]=base
    if args.support_force_goal_late_n is not None and t>=args.support_force_goal_late_start_s:
        wbc.support_palm_force_goal_n=args.support_force_goal_late_n
    if args.finger_forearm_cap_late_n is not None and t>=args.finger_forearm_cap_late_start_s:
        wbc.forearm_finger_force_limit_n=args.finger_forearm_cap_late_n
    if args.support_palm_position_weight_late is not None and t>=args.support_palm_position_weight_late_start_s:
        wbc.palm_position_weights[0]=args.support_palm_position_weight_late
    if args.pad_orientation_after_s is not None:
        if t>=args.pad_orientation_after_s and pad_hold_quat is None:
            pad_hold_quat=(start_pad_quat.copy() if args.pad_orientation_use_start else
                           data.qpos[pad_qadr+3:pad_qadr+7].copy())
        if pad_hold_quat is not None:wbc.qref[pad_qadr+3:pad_qadr+7]=pad_hold_quat
        wbc.weights[pad_vadr+3:pad_vadr+6]=(args.pad_pose_weight
            if t>=args.pad_orientation_after_s else original_pad_orientation_weights)
    if args.start_state_token=='947858479e10':
        wbc.qref[b_hold_qa]=state['qpos'][b_hold_qa]
        wbc.qref[b_hand_hold_qa]=state['qpos'][b_hand_hold_qa]
    wbc.qref[support_yaw_qa]+=args.support_wrist_yaw_offset_rad*min(1.,max(0.,t-args.support_wrist_yaw_start_s)/.5)
    wbc.qref[support_yaw_qa]+=args.support_wrist_yaw_late_offset_rad*min(1.,max(0.,t-args.support_wrist_yaw_late_start_s)/.5)
    wbc.qref[support_pinky_qa]+=args.support_pinky_flex_late_offset_rad*min(1.,max(0.,t-args.support_pinky_late_start_s)/.5)
    if any(args.support_palm_shift_late_m) or any(args.support_palm_shift_second_m) or any(args.support_palm_shift_third_m):
        support_ref.qpos[:]=wbc.qref;mujoco.mj_forward(model,support_ref)
        jp=np.zeros((3,model.nv));mujoco.mj_jacBody(model,support_ref,jp,None,left_palm)
        jac=jp[:,left_va]
        blend=min(1.,max(0.,t-args.support_palm_shift_start_s)/.5)
        blend2=min(1.,max(0.,t-args.support_palm_shift_second_start_s)/.5)
        blend3=min(1.,max(0.,t-args.support_palm_shift_third_start_s)/.5)
        target=(blend*np.asarray(args.support_palm_shift_late_m)+
                blend2*np.asarray(args.support_palm_shift_second_m)+
                blend3*np.asarray(args.support_palm_shift_third_m))
        delta=jac.T@np.linalg.solve(jac@jac.T+1e-4*np.eye(3),target)
        wbc.qref[left_qa]+=np.clip(delta,-.12,.12)
    wbc.qref[elbow.qposadr[0]]=.25
    if args.support_follow_gain and t>=args.support_follow_start_s:
        support_ref.qpos[:]=wbc.qref;mujoco.mj_forward(model,support_ref)
        live_rot=data.xmat[forearm_body].reshape(3,3)
        palm_rot=support_ref.xmat[left_palm].reshape(3,3)
        pos_error=data.xpos[forearm_body]+live_rot@relative_palm_pos-goal_palm_pos
        target_rot=(live_rot@goal_forearm_rot.T)@palm_rot
        rot_error=.5*sum((np.cross(palm_rot[:,i],target_rot[:,i]) for i in range(3)),start=np.zeros(3))
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacBody(model,support_ref,jp,jr,left_palm)
        jac=np.vstack((jp[:,left_va],jr[:,left_va]))
        delta=jac.T@np.linalg.solve(jac@jac.T+.02**2*np.eye(6),np.r_[pos_error,rot_error])
        wbc.qref[left_qa]+=args.support_follow_gain*np.clip(delta,-.12,.12)
        for j,qa in zip(left_joints,left_qa):
            lo,hi=model.jnt_range[j];wbc.qref[qa]=np.clip(wbc.qref[qa],lo+.03,hi-.03)
    if args.track_pad_pose:wbc.qref[pad_qadr+2]=base[pad_qadr+2]+planned
    if pregrasp_candidate is not None:
        blend=min(1.,t/args.index_preposition_s)
        for name,info in digit_info.items():
            if name=='middle' and args.middle_after_s is not None:continue
            if name=='index' or side_candidate is not None or middle_candidate is not None:
                qa=info['qa']
                wbc.qref[qa]=(1-blend)*base[qa]+blend*pregrasp_candidate[qa]
    if middle_candidate is not None and args.middle_after_s is not None:
        info=digit_info['middle'];qa=info['qa']
        blend=np.clip((t-args.middle_after_s)/.3,0.,1.)
        wbc.qref[qa]=(1-blend)*base[qa]+blend*middle_candidate[qa]
    ref.qpos[:]=base;mujoco.mj_forward(model,ref)
    jp=np.zeros((3,model.nv));mujoco.mj_jac(model,ref,jp,None,ref.xpos[wrist],wrist)
    J=jp[:,arm_va]
    correction=J.T@np.linalg.solve(J@J.T+1e-4*np.eye(3),
                                   np.array([args.wrist_x_shift_m*motion_u,0.,planned]))
    wbc.qref[arm_qa]+=np.clip(correction,-.12,.12)
    if any(args.pickup_arm_offsets_rad):
        arm_blend=np.clip((t-args.pickup_wrist_rotate_start_s)/.6,0.,1.)
        wbc.qref[arm_qa]+=arm_blend*np.asarray(args.pickup_arm_offsets_rad)
    if args.late_tab_pinch_after_s is not None and any(args.late_arm_offsets_rad):
        late_blend=np.clip((t-args.late_tab_pinch_after_s)/.6,0.,1.)
        wbc.qref[arm_qa]+=late_blend*np.asarray(args.late_arm_offsets_rad)
    wrist_rotate=np.clip((t-args.pickup_wrist_rotate_start_s)/.6,0.,1.)
    if wrist_rotate:
        for joint_name,offset in (('g1_a/right_wrist_pitch_joint',args.pickup_wrist_pitch_offset_rad),
                                  ('g1_a/right_wrist_yaw_joint',args.pickup_wrist_yaw_offset_rad)):
            qa=model.joint(joint_name).qposadr[0]
            wbc.qref[qa]+=wrist_rotate*offset
    for info in spool_digit_info.values():
        body=info['body'];rot_spool=data.xmat[spool].reshape(3,3)
        target=data.xpos[spool]+rot_spool@info['local']
        jp=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jp,None,data.xpos[body],body)
        jac=jp[:,info['va']]
        delta=np.linalg.solve(jac.T@jac+1e-3*np.eye(len(info['va'])),jac.T@(.6*(target-data.xpos[body])))
        wbc.qref[info['qa']]+=np.clip(delta,-.08,.08)
    rot=data.xmat[pad].reshape(3,3)
    if (args.transport_grasp_after_s is not None and t>=args.transport_grasp_after_s and
            transport_anchor is None and
            (not args.contact_gated_transport or transport_ready_streak>=5)):
        transport_anchor=(data.xpos[pad].copy(),rot.copy(),float(planned),float(motion_u))
        for info in digit_info.values():
            info['local']=rot.T@(data.xpos[info['body']]-data.xpos[pad])
    if args.freeze_digit_frame_after_s is not None and t>=args.freeze_digit_frame_after_s and fixed_digit_rot is None:
        fixed_digit_rot=rot.copy()
    for name,info in digit_info.items():
        if name=='middle' and args.middle_after_s is not None and t<args.middle_after_s+.3:continue
        if pregrasp_candidate is not None and (name=='index' or side_candidate is not None or middle_candidate is not None):
            if t<args.index_preposition_s:continue
            if not local_captured[name]:
                info['local']=rot.T@(data.xpos[info['body']]-data.xpos[pad])
                local_captured[name]=True
        body=info['body'];jp=np.zeros((3,model.nv))
        mujoco.mj_jac(model,data,jp,None,data.xpos[body],body)
        jac=jp[:,info['va']]
        local=info['local'].copy()
        if info is digit_info['index'] and args.index_closure_m:
            closure=args.index_closure_m*min(1.,max(0.,t-args.index_closure_after_s)/.3)
            local-=closure*local/max(1e-6,np.linalg.norm(local))
        if info is digit_info['thumb'] and args.thumb_closure_m:
            closure=args.thumb_closure_m*min(1.,max(0.,t-args.thumb_closure_after_s)/.3)
            local-=closure*local/max(1e-6,np.linalg.norm(local))
        if transport_anchor is None:
            target=data.xpos[pad]+(fixed_digit_rot if fixed_digit_rot is not None else rot)@local
        else:
            anchor_pos,anchor_rot,anchor_lift,anchor_u=transport_anchor
            target=anchor_pos+np.array([args.wrist_x_shift_m*(motion_u-anchor_u),0.,planned-anchor_lift])+anchor_rot@local
        if name=='index' and any(args.index_tip_offset_m):
            blend=np.clip((t-args.index_tip_offset_start_s)/.5,0.,1.)
            target+=blend*np.asarray(args.index_tip_offset_m)
        if name=='thumb' and any(args.thumb_tip_offset_m):
            blend=np.clip((t-args.thumb_tip_offset_start_s)/.5,0.,1.)
            target+=blend*np.asarray(args.thumb_tip_offset_m)
        delta=np.linalg.solve(jac.T@jac+1e-3*np.eye(jac.shape[1]),jac.T@(.6*(target-data.xpos[body])))
        wbc.qref[info['qa']]+=np.clip(delta,-.08,.08)
    if args.index_ik_target_rad is not None and transport_anchor is None:
        blend=np.clip((t-args.pickup_wrist_rotate_start_s)/.6,0.,1.)
        qa=digit_info['index']['qa']
        wbc.qref[qa]=(1.-blend)*wbc.qref[qa]+blend*np.asarray(args.index_ik_target_rad)
    if args.thumb_ik_target_rad is not None and transport_anchor is None:
        blend=np.clip((t-args.pickup_wrist_rotate_start_s)/.6,0.,1.)
        qa=digit_info['thumb']['qa']
        wbc.qref[qa]=(1.-blend)*wbc.qref[qa]+blend*np.asarray(args.thumb_ik_target_rad)
    if args.late_tab_pinch_after_s is not None:
        late_blend=np.clip((t-args.late_tab_pinch_after_s)/.6,0.,1.)
        for name,target in (('thumb',args.late_thumb_ik_target_rad),('index',args.late_index_ik_target_rad)):
            if target is not None:
                qa=digit_info[name]['qa']
                wbc.qref[qa]=(1.-late_blend)*wbc.qref[qa]+late_blend*np.asarray(target)
    if (args.online_tab_ik_after_s is not None and t>=args.online_tab_ik_after_s and
            (transport_anchor is None or args.online_tab_ik_transport)):
      if tip_first_target is not None:
        blend=np.clip((t-args.online_tab_ik_after_s)/.6,0.,1.)
        wbc.qref[ik_qa]=(1.-blend)*data.qpos[ik_qa]+blend*tip_first_target
      else:
        ik_data.qpos[:]=wbc.qref
        ik_data.qpos[pad_qadr:pad_qadr+7]=data.qpos[pad_qadr:pad_qadr+7]
        mujoco.mj_forward(model,ik_data)
        tab_R=ik_data.geom_xmat[tab_geom].reshape(3,3)
        tab_pos=ik_data.geom_xpos[tab_geom].copy()
        tab_pos[2]+=unload_offset
        if transport_anchor is not None:
            tab_pos[2]+=planned-transport_anchor[2]
        if args.online_tab_up_force_feedback_gain and rows:
            recent=rows[-5:]
            measured_up=float(np.median([sum(v for k,v in r['actual_pad_up_force_by_body_n'].items()
                                             if k.startswith('g1_a/wuji_right/')) for r in recent]))
            desired_correction=np.clip(args.online_tab_up_force_feedback_gain*
                (args.online_tab_up_force_target_n-measured_up),-.008,.008)
            tab_feedback_z=.8*tab_feedback_z+.2*desired_correction
        effective_tab_target_z=args.online_tab_target_z_m+tab_feedback_z
        thumb_offset=(args.online_tab_thumb_offset_m if args.online_tab_thumb_offset_m is not None
                      else args.online_tab_side_offset_m)
        index_offset=(args.online_tab_index_offset_m if args.online_tab_index_offset_m is not None
                      else args.online_tab_side_offset_m)
        tab_targets=(tab_pos+tab_R[:,1]*thumb_offset+tab_R[:,2]*effective_tab_target_z,
                     tab_pos-tab_R[:,1]*index_offset+tab_R[:,2]*effective_tab_target_z)
        if args.online_tab_middle:
            middle_target=tab_targets[1]+tab_R[:,0]*.010
        thumb_id=digit_info['thumb']['body'];index_id=digit_info['index']['body']
        if args.online_tab_tip_task_weight:
            wbc.tip_position_weight=args.online_tab_tip_task_weight
            wbc.tip_position_targets={model.body(thumb_id).name:tab_targets[0],
                                      model.body(index_id).name:tab_targets[1]}
        wrist_start=ik_data.xpos[wrist].copy();ik_start=wbc.qref[ik_qa].copy()
        def tab_residual(x):
            ik_data.qpos[ik_qa]=x;mujoco.mj_forward(model,ik_data)
            parts=[ik_data.xpos[thumb_id]-tab_targets[0],
                   ik_data.xpos[index_id]-tab_targets[1]]
            if args.online_tab_middle:
                parts.append(ik_data.xpos[digit_info['middle']['body']]-middle_target)
            if args.online_tab_middle_clearance_m is not None:
                middle_body=model.body('g1_a/wuji_right/r_index_finger_middle').id
                proximal_body=model.body('g1_a/wuji_right/r_index_finger_proximal_abd').id
                middle_radius=np.linalg.norm(ik_data.xpos[middle_body,:2]-tab_pos[:2])
                proximal_radius=np.linalg.norm(ik_data.xpos[proximal_body,:2]-tab_pos[:2])
                parts.append(np.array([2.*max(0.,args.online_tab_middle_clearance_m-middle_radius),
                                       2.*max(0.,.03-proximal_radius)]))
            parts.extend((args.online_tab_wrist_hold_weight*(ik_data.xpos[wrist]-wrist_start),
                          .002*(x[:len(arm_qa)]-ik_start[:len(arm_qa)])))
            return np.concatenate(parts)
        fit=least_squares(tab_residual,np.clip(ik_start,ik_lo,ik_hi),bounds=(ik_lo,ik_hi),
                          max_nfev=35)
        wbc.qref[ik_qa]=fit.x
        ref.qpos[:]=wbc.qref;mujoco.mj_forward(model,ref)
        online_ik_diagnostic={
            'fit_thumb_target_error_m':float(np.linalg.norm(fit.fun[:3])),
            'fit_index_target_error_m':float(np.linalg.norm(fit.fun[3:6])),
            'reference_thumb_target_error_m':float(np.linalg.norm(ref.xpos[thumb_id]-tab_targets[0])),
            'reference_index_target_error_m':float(np.linalg.norm(ref.xpos[index_id]-tab_targets[1])),
            'target_thumb_m':tab_targets[0].tolist(),
            'target_index_m':tab_targets[1].tolist()}
    try:
        data.ctrl[:],_,_=wbc.solve(data)
        predicted_dressing_up_n=wbc.last_predicted_dressing_up_force_n
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
                     'predicted_dressing_up_force_n':predicted_dressing_up_n,
                     'support_impulse_by_body_ns':impulses,
                     'feet_supported':feet=={'g1_a','g1_b'},'deep_robot_contact':deep,
                     'spool_grip_impulse_by_body_ns':spool_grip})
    rows[-1]['digit_tip_m']={name:data.xpos[info['body']].tolist() for name,info in digit_info.items()}
    rows[-1]['digit_joint_error_rad']={name:(data.qpos[info['qa']]-wbc.qref[info['qa']]).tolist()
                                         for name,info in digit_info.items()}
    rows[-1]['arm_joint_error_rad']=(data.qpos[arm_qa]-wbc.qref[arm_qa]).tolist()
    if online_ik_diagnostic is not None:
        online_ik_diagnostic['actual_thumb_target_error_m']=float(np.linalg.norm(
            data.xpos[digit_info['thumb']['body']]-online_ik_diagnostic['target_thumb_m']))
        online_ik_diagnostic['actual_index_target_error_m']=float(np.linalg.norm(
            data.xpos[digit_info['index']['body']]-online_ik_diagnostic['target_index_m']))
        rows[-1]['online_ik_diagnostic']=online_ik_diagnostic
    actual_up_by_source={}
    for ci,c in enumerate(data.contact):
        if c.geom1<0 or c.geom2<0:continue
        ga=model.geom(c.geom1).name;gb=model.geom(c.geom2).name
        if not any(name in (ga,gb) for name in ('dressing_pad_contact','dressing_pull_tab')):continue
        other_geom=c.geom2 if ga in ('dressing_pad_contact','dressing_pull_tab') else c.geom1
        other_body=model.body(model.geom_bodyid[other_geom]).name
        wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
        sign=-1. if ga in ('dressing_pad_contact','dressing_pull_tab') else 1.
        up=float((sign*c.frame.reshape(3,3).T@wrench[:3])[2])
        actual_up_by_source[other_body]=actual_up_by_source.get(other_body,0.)+up
    rows[-1]['actual_pad_up_force_by_body_n']=actual_up_by_source
    rows[-1]['planned_unload_m']=unload_offset
    rows[-1]['effective_online_tab_target_z_m']=effective_tab_target_z
    rows[-1]['pad_contacts']=[{
        'geoms':[model.geom(c.geom1).name,model.geom(c.geom2).name],
        'bodies':[model.body(model.geom_bodyid[c.geom1]).name,
                  model.body(model.geom_bodyid[c.geom2]).name],
        'distance_m':float(c.dist),
        'normal_force_n':contact_force(ci),
        'pad_to_other_normal_world':((1. if model.geom(c.geom1).name in ('dressing_pad_contact','dressing_pull_tab') else -1.)*
                                      np.asarray(c.frame[:3])).tolist()} for ci,c in enumerate(data.contact)
        if any(name in (model.geom(c.geom1).name,model.geom(c.geom2).name) for name in ('dressing_pad_contact','dressing_pull_tab'))]
    if args.contact_gated_transport:
        transport_gate_last=evaluate_pickup_gate(rows[-1])
        transport_ready_streak=(transport_ready_streak+1 if transport_gate_last.ready else 0)
        rows[-1]['transport_gate']={**vars(transport_gate_last),
                                    'consecutive_ready_frames':transport_ready_streak,
                                    'transport_started':transport_anchor is not None}
    lateral_limit=(args.max_pad_lateral_shift_m if args.max_pad_lateral_shift_m is not None
                   else max(.01,abs(args.wrist_x_shift_m)+.01))
    if rows[-1]['pad_lateral_shift_m']>lateral_limit:
        errors.append('pad_lateral_shift_over_plan_plus_10mm');break
    if rows[-1]['spool_shift_m']>.005:errors.append('spool_shift_over_5mm');break
report={'scope':'Checkpoint diagnostic; no full-sequence dressing pickup claim',
        'lift_m':args.lift_m,'duration_s':args.duration_s,
        'motion_end_s':motion_end,
        'smooth_motion':args.smooth_motion,
        'dressing_force_goal_n':args.dressing_force_goal_n,
        'dressing_thumb_goal_n':args.dressing_thumb_goal_n,
        'dressing_index_goal_n':args.dressing_index_goal_n,
        'index_closure_m':args.index_closure_m,
        'index_closure_after_s':args.index_closure_after_s,
        'thumb_closure_m':args.thumb_closure_m,
        'thumb_closure_after_s':args.thumb_closure_after_s,
        'freeze_digit_frame_after_s':args.freeze_digit_frame_after_s,
        'index_pregrasp_candidate':args.index_pregrasp_candidate,
        'side_pregrasp_candidate':args.side_pregrasp_candidate,
        'middle_support_candidate':args.middle_support_candidate,
        'middle_after_s':args.middle_after_s,
        'index_weight':args.index_weight,
        'pickup_wrist_position_weight':args.pickup_wrist_position_weight,
        'pickup_wrist_rotation_weight':args.pickup_wrist_rotation_weight,
        'pickup_wrist_pitch_offset_rad':args.pickup_wrist_pitch_offset_rad,
        'pickup_wrist_yaw_offset_rad':args.pickup_wrist_yaw_offset_rad,
        'pickup_wrist_rotate_start_s':args.pickup_wrist_rotate_start_s,
        'pickup_arm_offsets_rad':args.pickup_arm_offsets_rad,
        'index_ik_target_rad':args.index_ik_target_rad,
        'thumb_ik_target_rad':args.thumb_ik_target_rad,
        'late_tab_pinch_after_s':args.late_tab_pinch_after_s,
        'late_thumb_ik_target_rad':args.late_thumb_ik_target_rad,
        'late_index_ik_target_rad':args.late_index_ik_target_rad,
        'late_arm_offsets_rad':args.late_arm_offsets_rad,
        'online_tab_ik_after_s':args.online_tab_ik_after_s,
        'online_tab_target_z_m':args.online_tab_target_z_m,
        'online_tab_side_offset_m':args.online_tab_side_offset_m,
        'online_tab_thumb_offset_m':args.online_tab_thumb_offset_m,
        'online_tab_index_offset_m':args.online_tab_index_offset_m,
        'online_tab_middle_clearance_m':args.online_tab_middle_clearance_m,
        'online_tab_ik_transport':args.online_tab_ik_transport,
        'online_tab_middle':args.online_tab_middle,
        'online_tab_wrist_hold_weight':args.online_tab_wrist_hold_weight,
        'online_tab_up_force_feedback_gain':args.online_tab_up_force_feedback_gain,
        'online_tab_up_force_target_n':args.online_tab_up_force_target_n,
        'online_tab_index_dip_min_rad':args.online_tab_index_dip_min_rad,
        'online_tab_tip_task_weight':args.online_tab_tip_task_weight,
        'tip_first_candidate_index':args.tip_first_candidate_index,
        'transport_grasp_after_s':args.transport_grasp_after_s,
        'transport_extra_lift_m':args.transport_extra_lift_m,
        'unload_start_s':args.unload_start_s,
        'unload_lift_m':args.unload_lift_m,
        'unload_ramp_s':args.unload_ramp_s,
        'contact_gated_transport':args.contact_gated_transport,
        'transport_started':transport_anchor is not None,
        'final_transport_gate':({**vars(transport_gate_last),
                                 'consecutive_ready_frames':transport_ready_streak}
                                if transport_gate_last is not None else None),
        'pull_tab_scene':args.pull_tab_scene,
        'pinch_post_scene':args.pinch_post_scene,
        'index_tip_offset_m':args.index_tip_offset_m,
        'index_tip_offset_start_s':args.index_tip_offset_start_s,
        'thumb_tip_offset_m':args.thumb_tip_offset_m,
        'thumb_tip_offset_start_s':args.thumb_tip_offset_start_s,
        'dressing_lift_min_n':args.dressing_lift_min_n,
        'dressing_up_goal_n':args.dressing_up_goal_n,
        'dressing_up_goal_start_s':args.dressing_up_goal_start_s,
        'dressing_force_start_s':args.dressing_force_start_s,
        'dressing_force_end_s':args.dressing_force_end_s,
        'track_pad_pose':args.track_pad_pose,
        'track_pad_orientation':args.track_pad_orientation,
        'pad_pose_weight':args.pad_pose_weight,
        'pad_orientation_after_s':args.pad_orientation_after_s,
        'pad_orientation_use_start':args.pad_orientation_use_start,
        'wrist_x_shift_m':args.wrist_x_shift_m,
        'max_pad_lateral_shift_m':args.max_pad_lateral_shift_m,
        'pad_lateral_feedback_gain':args.pad_lateral_feedback_gain,
        'pad_lateral_feedback_start_s':args.pad_lateral_feedback_start_s,
        'support_wrist_yaw_offset_rad':args.support_wrist_yaw_offset_rad,
        'support_wrist_yaw_start_s':args.support_wrist_yaw_start_s,
        'support_wrist_yaw_late_offset_rad':args.support_wrist_yaw_late_offset_rad,
        'support_wrist_yaw_late_start_s':args.support_wrist_yaw_late_start_s,
        'support_follow_gain':args.support_follow_gain,
        'support_follow_start_s':args.support_follow_start_s,
        'support_pinky_flex_late_offset_rad':args.support_pinky_flex_late_offset_rad,
        'support_pinky_late_start_s':args.support_pinky_late_start_s,
        'support_palm_shift_late_m':args.support_palm_shift_late_m,
        'support_palm_shift_start_s':args.support_palm_shift_start_s,
        'support_palm_shift_second_m':args.support_palm_shift_second_m,
        'support_palm_shift_second_start_s':args.support_palm_shift_second_start_s,
        'support_palm_shift_third_m':args.support_palm_shift_third_m,
        'support_palm_shift_third_start_s':args.support_palm_shift_third_start_s,
        'support_palm_position_weight_late':args.support_palm_position_weight_late,
        'support_palm_position_weight_late_start_s':args.support_palm_position_weight_late_start_s,
        'elbow_hold_weight':args.elbow_hold_weight,
        'support_force_goal_late_n':args.support_force_goal_late_n,
        'support_force_goal_late_start_s':args.support_force_goal_late_start_s,
        'finger_forearm_cap_late_n':args.finger_forearm_cap_late_n,
        'finger_forearm_cap_late_start_s':args.finger_forearm_cap_late_start_s,
        'tag':args.tag,
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
if args.duration_s!=3.:stem+=f'_duration{args.duration_s:g}'
if args.index_closure_after_s:stem+=f'_icloseafter{args.index_closure_after_s:g}'
if args.thumb_closure_m:stem+=f'_tclose{args.thumb_closure_m:g}_after{args.thumb_closure_after_s:g}'
if args.freeze_digit_frame_after_s is not None:stem+=f'_fixedframe{args.freeze_digit_frame_after_s:g}'
if args.middle_support_candidate>=0:stem+=f'_middle{args.middle_support_candidate}'
if args.middle_after_s is not None:stem+=f'_middleafter{args.middle_after_s:g}'
if args.motion_end_s is not None:stem+=f'_motionend{args.motion_end_s:g}'
if args.smooth_motion:stem+='_smooth'
if args.dressing_force_end_s is not None:stem+=f'_forceend{args.dressing_force_end_s:g}'
if args.track_pad_pose:stem+=f'_pose{args.pad_pose_weight:g}'
if args.track_pad_orientation:stem+=f'_orient{args.pad_pose_weight:g}'
if args.pad_orientation_after_s is not None:stem+=f'_orientafter{args.pad_orientation_after_s:g}w{args.pad_pose_weight:g}'
if args.wrist_x_shift_m:stem+=f'_xshift{args.wrist_x_shift_m:g}'
if args.support_wrist_yaw_offset_rad:stem+=f'_syaw{args.support_wrist_yaw_offset_rad:g}_at{args.support_wrist_yaw_start_s:g}'
if args.support_wrist_yaw_late_offset_rad:stem+=f'_lateyaw{args.support_wrist_yaw_late_offset_rad:g}_at{args.support_wrist_yaw_late_start_s:g}'
if args.support_follow_gain:stem+=f'_follow{args.support_follow_gain:g}_at{args.support_follow_start_s:g}'
if args.support_pinky_flex_late_offset_rad:stem+=f'_pinky{args.support_pinky_flex_late_offset_rad:g}_at{args.support_pinky_late_start_s:g}'
if any(args.support_palm_shift_late_m):stem+='_palmshift'+','.join(f'{x:g}' for x in args.support_palm_shift_late_m)
if any(args.support_palm_shift_second_m):stem+='_palmshift2'+','.join(f'{x:g}' for x in args.support_palm_shift_second_m)+f'_at{args.support_palm_shift_second_start_s:g}'
if any(args.support_palm_shift_third_m):stem+='_palmshift3'+','.join(f'{x:g}' for x in args.support_palm_shift_third_m)+f'_at{args.support_palm_shift_third_start_s:g}'
if args.support_palm_position_weight_late is not None:stem+=f'_palmweight{args.support_palm_position_weight_late:g}_at{args.support_palm_position_weight_late_start_s:g}'
if args.elbow_hold_weight!=1.:stem+=f'_elbowweight{args.elbow_hold_weight:g}'
if args.support_force_goal_late_n is not None:stem+=f'_supportforce{args.support_force_goal_late_n:g}_at{args.support_force_goal_late_start_s:g}'
if args.tag:stem+=f'_tag{args.tag}'
stem+=f'_from{args.start_state_token}'
if args.start_state_token=='947858479e10':stem+=f'_bmin{args.b_hold_min_force_n:g}_bgoal{args.b_hold_force_goal_n:g}'
if len(str(OUT/f'{stem}.json'))>240:
    stem='joint_dressing_pilot_'+hashlib.sha256(stem.encode('utf8')).hexdigest()[:16]
report['artifact_stem']=stem
(OUT/f'{stem}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
np.savez_compressed(OUT/f'{stem}_final_state.npz',qpos=data.qpos,qvel=data.qvel)
np.savez_compressed(OUT/f'{stem}_qpos.npz',qpos=np.asarray(recorded_qpos),
                    time_s=np.asarray([row['time_s'] for row in rows]))
print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
