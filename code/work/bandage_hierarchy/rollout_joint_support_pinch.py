"""Dynamic double-G1 WBC screen for forearm support, pad pinch, and optional spool grasp."""
import json
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--staged',action='store_true',help='Use contact-free pregrasp path before coordinated finger closure')
parser.add_argument('--grasp-finger-weight',type=float,default=1.,help='WBC acceleration tracking weight for A-right thumb and index')
parser.add_argument('--palm-rotation-weight',type=float,default=10.,help='WBC wrist orientation tracking weight')
parser.add_argument('--right-palm-position-weight',type=float,default=1000.,help='WBC A-right wrist position tracking weight')
parser.add_argument('--b-palm-position-weight',type=float,default=1000.,help='WBC B-left wrist position tracking weight')
parser.add_argument('--b-finger-weight',type=float,default=1.,help='WBC B-left thumb and index acceleration tracking weight')
parser.add_argument('--b-analytic-grip',action='store_true',help='Track live spool pose with B-left digit Jacobian feedback')
parser.add_argument('--analytic-grip-specialist',action='store_true',
                    help='Route pad-relative finger corrections after support and pad touch')
parser.add_argument('--hold-s',type=float,default=1.5,help='Final pinch hold duration')
parser.add_argument('--support-finger-force-cap-n',type=float,default=2.,
                    help='Predicted A-left finger contact force cap on the forearm')
parser.add_argument('--support-total-force-cap-n',type=float,default=12.,
                    help='Predicted total A-left contact force cap on the forearm')
parser.add_argument('--spool-after',action='store_true',
                    help='Use the separate B-root x=0.55 m layout and append B-left spool approach')
parser.add_argument('--settled-spool',action='store_true',help='Use z=0.702 m resting spool state and matched B-left path')
parser.add_argument('--wbc-backend',choices=('CLARABEL','DIRECT','OSQP'),default='CLARABEL')
args=parser.parse_args()
if args.settled_spool and not args.spool_after:parser.error('Settled spool requires --spool-after')
if args.b_analytic_grip and not args.settled_spool:parser.error('B analytic grip requires --settled-spool')
if not 1.<=args.grasp_finger_weight<=30.:
    parser.error('Grasp finger weight must be in [1,30]')
if not 1.<=args.palm_rotation_weight<=300.:
    parser.error('Palm rotation weight must be in [1,300]')
if not 1000.<=args.right_palm_position_weight<=10000.:
    parser.error('A-right palm position weight must be in [1000,10000]')
if not 1000.<=args.b_palm_position_weight<=10000.:
    parser.error('B-left palm position weight must be in [1000,10000]')
if not 1.<=args.b_finger_weight<=30.:
    parser.error('B-left finger weight must be in [1,30]')
if args.analytic_grip_specialist and not args.staged:
    parser.error('Analytic grip specialist requires staged pregrasp path')
if not .5<=args.hold_s<=5.:
    parser.error('Hold duration must be in [0.5,5] seconds')
if not .2<=args.support_finger_force_cap_n<=2.:
    parser.error('Support finger force cap must be in [0.2,2] N')
if not 5.<=args.support_total_force_cap_n<=15.:
    parser.error('Support total force cap must be in [5,15] N')
sys.path.insert(0, str(ROOT / 'work/wuji_multiagent'))
from joint_wbc import JointWBC
from support_quality import SupportQualityMonitor, SupportSample
sys.path.insert(0,str(ROOT/'work/bandage_hierarchy'))
from controller_router import HandExpertRouter, HandProposal, Observation, Mode

model_name=('joint_spool_grasp_static_z0p73_root0p55_spoolz0p702_variant.xml' if args.settled_spool else
            'joint_spool_grasp_static_z0p745_root0p55_variant.xml' if args.spool_after else
            'joint_support_pinch_variant.xml')
model = mujoco.MjModel.from_xml_path(str(OUT / model_name))
data = mujoco.MjData(model)
left_path = np.load(OUT / 'dynamic_palm_support_rrt.npz')['qpos'].copy()
right_path = np.load(OUT / ('joint_support_pregrasp_rrt.npz' if args.staged
                            else 'joint_support_pinch_rrt.npz'))['qpos'].copy()
if args.spool_after:
    b_root_qadr=model.joint('g1_b/floating_base').qposadr[0]
    left_path[:,b_root_qadr]=.55
    right_path[:,b_root_qadr]=.55
    if args.settled_spool:
        spool_z_qadr=model.joint('bandage_spool_free').qposadr[0]+2
        left_path[:,spool_z_qadr]=.702
        right_path[:,spool_z_qadr]=.702
if np.max(np.abs(left_path[-1] - right_path[0])) > 1e-9:
    raise RuntimeError('Support and pinch paths do not join')
path = np.vstack([left_path, right_path[1:]])
if args.staged:
    closure=np.load(OUT/'joint_support_pinch_static_candidate.npz')['qpos'].copy()
    if args.spool_after:closure[b_root_qadr]=.55
    if args.settled_spool:closure[spool_z_qadr]=.702
    path=np.vstack([path,closure])
if args.spool_after:
    spool_path=np.load(OUT/('joint_spool_grasp_rrt_settled.npz' if args.settled_spool
                            else 'joint_spool_grasp_rrt.npz'))['qpos']
    if np.max(np.abs(path[-1]-spool_path[0]))>1e-9:
        raise RuntimeError('Pinch and spool paths do not join')
    path=np.vstack([path,spool_path[1:]])
if args.spool_after:
    static_deep=[]
    for segment,(a,b) in enumerate(zip(path,path[1:]),start=1):
        samples=max(2,int(np.ceil(np.linalg.norm(b-a)/.04)))
        for u in np.linspace(0,1,samples+1):
            data.qpos[:]=(1-u)*a+u*b
            mujoco.mj_forward(model,data)
            for c in data.contact:
                if c.geom1<0 or c.geom2<0 or c.dist>=-.003:continue
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                        model.body(model.geom_bodyid[c.geom2]).name)
                if 'floor' not in geoms and any(body.startswith(('g1_a/','g1_b/')) for body in bodies):
                    static_deep.append({'segment':segment,'fraction':float(u),
                                        'bodies':bodies,'distance_m':float(c.dist)})
                    break
    if static_deep:
        raise RuntimeError(f'Combined spool path has {len(static_deep)} static deep-contact samples; first {static_deep[0]}')
data.qpos[:] = path[0]
mujoco.mj_forward(model, data)
palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
def contact_filter(m, c):
    return any(m.body(m.geom_bodyid[int(g)]).name.startswith(('g1_a/', 'g1_b/'))
               for g in c.geom if g >= 0)
wbc = JointWBC(model, data.qpos.copy(), palm_names=palms, contact_filter=contact_filter,
               palm_rotation_weights=[10.,args.palm_rotation_weight,10.,10.],
               palm_position_weights=[1000.,args.right_palm_position_weight,args.b_palm_position_weight,1000.])
wbc.backend=args.wbc_backend
wbc.forearm_contact_force_limit_n = args.support_total_force_cap_n
wbc.forearm_finger_force_limit_n = args.support_finger_force_cap_n
grasp_vadr=np.array([model.jnt_dofadr[j] for j in range(model.njnt)
                    if model.joint(j).name.startswith(('g1_a/wuji_right/r_thumb_',
                                                        'g1_a/wuji_right/r_index_finger_'))],dtype=int)
wbc.weights[grasp_vadr]=args.grasp_finger_weight
b_grasp_vadr=np.array([model.jnt_dofadr[j] for j in range(model.njnt)
                       if model.joint(j).name.startswith(('g1_b/wuji_left/l_thumb_',
                                                           'g1_b/wuji_left/l_index_finger_'))],dtype=int)
wbc.weights[b_grasp_vadr]=args.b_finger_weight
elbow_joint = model.joint('training_elbow_hinge').id
elbow_qadr = model.jnt_qposadr[elbow_joint]
elbow_vadr = model.jnt_dofadr[elbow_joint]
forearm_body = model.body('training_forearm_dynamic').id
forearm_geom = model.geom('manikin_forearm').id
left_palm = model.body('g1_a/wuji_left/l_wrist').id
left_joints = [j for j in range(model.njnt)
               if model.joint(j).name.startswith('g1_a/left_')
               and any(s in model.joint(j).name for s in ('shoulder','elbow','wrist'))
               and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
left_qadr = np.array([model.jnt_qposadr[j] for j in left_joints], dtype=int)
left_vadr = np.array([model.jnt_dofadr[j] for j in left_joints], dtype=int)
ref = mujoco.MjData(model)
ref.qpos[:] = left_path[-1]
mujoco.mj_forward(model, ref)
goal_forearm_pos = ref.xpos[forearm_body].copy()
goal_forearm_rot = ref.xmat[forearm_body].reshape(3,3).copy()
goal_palm_pos = ref.xpos[left_palm].copy()
goal_palm_rot = ref.xmat[left_palm].reshape(3,3).copy()
relative_palm_pos = goal_forearm_rot.T @ (goal_palm_pos - goal_forearm_pos)
def follow_forearm(qref):
    qref[elbow_qadr] = data.qpos[elbow_qadr]
    ref.qpos[:] = qref
    mujoco.mj_forward(model, ref)
    live_rot = data.xmat[forearm_body].reshape(3,3)
    palm_rot = ref.xmat[left_palm].reshape(3,3)
    pos_error = data.xpos[forearm_body] + live_rot @ relative_palm_pos - goal_palm_pos
    target_rot = (live_rot @ goal_forearm_rot.T) @ palm_rot
    rot_error = .5 * sum((np.cross(palm_rot[:,i], target_rot[:,i]) for i in range(3)),
                         start=np.zeros(3))
    jp = np.zeros((3,model.nv)); jr = np.zeros_like(jp)
    mujoco.mj_jacBody(model,ref,jp,jr,left_palm)
    J = np.vstack((jp[:,left_vadr],jr[:,left_vadr]))
    delta = J.T @ np.linalg.solve(J@J.T + .02**2*np.eye(6), np.r_[pos_error,rot_error])
    qref[left_qadr] += .5*np.clip(delta,-.12,.12)
    for j,a in zip(left_joints,left_qadr):
        lo,hi=model.jnt_range[j]
        qref[a]=np.clip(qref[a],lo+.03,hi-.03)

hz = 50
interval = round(1/hz/model.opt.timestep)
if abs(interval*model.opt.timestep-1/hz)>1e-9:
    raise ValueError('Simulation step does not divide 50 Hz')
durations = [1.2] + [.8]*(len(path)-2) + [args.hold_s]
schedule = []
for segment, duration in enumerate(durations[:-1],start=1):
    schedule.extend((segment,k+1,round(duration*hz)) for k in range(round(duration*hz)))
schedule.extend((len(path)-1,round(durations[-2]*hz),round(durations[-2]*hz))
                for _ in range(round(durations[-1]*hz)))
support_monitor = SupportQualityMonitor()
trace=[]
errors=[]
foot_steps=0
deep_steps=0
peak_support_force=0.
peak_thumb_force=0.
peak_index_force=0.
support_valid_steps=0
both_digits_opposed_steps=0
opposed_run=0
longest_opposed_run=0
pad_body=model.body('dressing_pad').id
spool_body=model.body('bandage_spool').id
spool_start=data.xpos[spool_body].copy()
max_spool_displacement=0.
peak_b_thumb_force=0.
peak_b_index_force=0.
b_opposed_run=0
b_longest_opposed_run=0
thumb_tip=model.body('g1_a/wuji_right/r_thumb_tip_sensor_frame').id
index_tip=model.body('g1_a/wuji_right/r_index_finger_tip_sensor_frame').id
right_arm_qadr=np.array([model.jnt_qposadr[j] for j in range(model.njnt)
                         if model.joint(j).name.startswith('g1_a/right_')
                         and any(s in model.joint(j).name for s in ('shoulder','elbow','wrist'))],dtype=int)
right_digit_qadr=np.array([model.jnt_qposadr[j] for j in range(model.njnt)
                           if model.joint(j).name.startswith(('g1_a/wuji_right/r_thumb_',
                                                               'g1_a/wuji_right/r_index_finger_'))],dtype=int)
pad_start=data.xpos[pad_body].copy()
max_pad_displacement=0.
first_pad_contact_time_s=None
worst_deep_contact=None
worst_penetration_m=0.
hand_qadr=np.array([model.jnt_qposadr[model.actuator_trnid[i,0]] for i in wbc.hand_indices],dtype=int)
router=HandExpertRouter(len(hand_qadr))
router_modes={mode.value:0 for mode in Mode}
first_specialist_time_s=None
last_foot_supported=True
last_left_contact=False
last_right_contact=False
goal_data=mujoco.MjData(model)
goal_data.qpos[:]=path[-1]
mujoco.mj_forward(model,goal_data)
goal_pad_rot=goal_data.xmat[pad_body].reshape(3,3)
digit_info={}
for digit,bid in (('thumb',thumb_tip),('index',index_tip)):
    joints=[j for j in range(model.njnt)
            if model.joint(j).name.startswith(f'g1_a/wuji_right/r_{digit}_')]
    digit_info[digit]={
        'qadr':np.array([model.jnt_qposadr[j] for j in joints],dtype=int),
        'vadr':np.array([model.jnt_dofadr[j] for j in joints],dtype=int),
        'body':bid,
        'pad_local':goal_pad_rot.T@(goal_data.xpos[bid]-goal_data.xpos[pad_body])}
b_digit_info={}
if args.b_analytic_grip:
    goal_spool_rot=goal_data.xmat[spool_body].reshape(3,3)
    for digit in ('thumb','index'):
        bid=model.body(f'g1_b/wuji_left/l_{digit if digit=="thumb" else "index_finger"}_tip_sensor_frame').id
        joints=[j for j in range(model.njnt)
                if model.joint(j).name.startswith(f'g1_b/wuji_left/l_{digit if digit=="thumb" else "index_finger"}_')]
        local=goal_spool_rot.T@(goal_data.xpos[bid]-goal_data.xpos[spool_body])
        local[1]+=(-.008 if digit=='thumb' else .005)
        b_digit_info[digit]={'body':bid,'qadr':np.array([model.jnt_qposadr[j] for j in joints],dtype=int),
                             'vadr':np.array([model.jnt_dofadr[j] for j in joints],dtype=int),
                             'spool_local':local}
for segment,k,count in schedule:
    u=(k/count)**2*(3-2*k/count)
    wbc.qref[:]=(1-u)*path[segment-1]+u*path[segment]
    if segment>=2:
        follow_forearm(wbc.qref)
    if b_digit_info and segment>=len(path)-2:
        spool_rot=data.xmat[spool_body].reshape(3,3)
        for info in b_digit_info.values():
            bid=info['body'];vadr=info['vadr'];qadr=info['qadr']
            target=data.xpos[spool_body]+spool_rot@info['spool_local']
            jac=np.zeros((3,model.nv))
            mujoco.mj_jac(model,data,jac,None,data.xpos[bid],bid)
            J=jac[:,vadr]
            delta=np.linalg.solve(J.T@J+1e-3*np.eye(len(vadr)),J.T@(.6*(target-data.xpos[bid])))
            for address,offset in zip(qadr,np.clip(delta,-.08,.08)):
                wbc.qref[address]+=offset
    if args.analytic_grip_specialist:
        base_fingers=wbc.qref[hand_qadr].copy()
        proposed=base_fingers.copy()
        pad_rot=data.xmat[pad_body].reshape(3,3)
        for digit,info in digit_info.items():
            bid=info['body'];vadr=info['vadr'];qadr=info['qadr']
            target=data.xpos[pad_body]+pad_rot@info['pad_local']
            jac=np.zeros((3,model.nv))
            mujoco.mj_jac(model,data,jac,None,data.xpos[bid],bid)
            J=jac[:,vadr]
            delta=np.linalg.solve(J.T@J+1e-3*np.eye(len(vadr)),J.T@(.6*(target-data.xpos[bid])))
            for address,offset in zip(qadr,np.clip(delta,-.08,.08)):
                matching=np.flatnonzero(hand_qadr==address)
                if len(matching)==1:proposed[matching[0]]+=offset
        phase='pinch_dressing_supported' if segment==len(path)-1 and k==count else 'approach'
        decision=router.step(Observation(
            time_s=float(data.time),phase=phase,body_valid=True,
            feet_supported=last_foot_supported,
            arm_pose_stable=bool(np.max(np.abs(data.qvel[right_arm_qadr]))<.5),
            hand_contacts=(last_left_contact,last_right_contact,False,False),
            hand_feedback_fresh=True,dressing_grasp_valid=False,
            forearm_support_valid=bool(support_monitor.last_report.get('valid',False))),
            base_fingers,HandProposal(float(data.time),proposed,.95))
        router_modes[decision.mode.value]+=1
        if decision.mode==Mode.HAND and first_specialist_time_s is None:
            first_specialist_time_s=float(data.time)
        wbc.qref[hand_qadr]=decision.finger_targets_rad
    try:
        data.ctrl[:],_,_=wbc.solve(data)
        frame_impulse={}
        frame_peak=0.
        frame_foot=True
        frame_deep=False
        thumb_force=0.; index_force=0.
        normals={'thumb':np.zeros(3),'index':np.zeros(3)}
        for _ in range(interval):
            mujoco.mj_step(model,data)
            supported=set()
            thumb_force=0.;index_force=0.
            normals={'thumb':np.zeros(3),'index':np.zeros(3)}
            b_forces={'thumb':0.,'index':0.}
            b_normals={'thumb':np.zeros(3),'index':np.zeros(3)}
            support_force=0.
            for ci,c in enumerate(data.contact):
                if c.geom1<0 or c.geom2<0:continue
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                        model.body(model.geom_bodyid[c.geom2]).name)
                if 'floor' in geoms:
                    for robot in ('g1_a','g1_b'):
                        if any(b.startswith(robot+'/') for b in bodies):supported.add(robot)
                elif any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
                    frame_deep=True
                    if -float(c.dist)>worst_penetration_m:
                        worst_penetration_m=-float(c.dist)
                        worst_deep_contact={'time_s':float(data.time),'bodies':bodies,'geoms':geoms,
                                            'distance_m':float(c.dist)}
                if 'manikin_forearm' in geoms:
                    hand=next((b for b in bodies if b.startswith('g1_a/wuji_left/')),None)
                    if hand:
                        wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
                        force=max(0.,float(wrench[0]))
                        frame_impulse[hand]=frame_impulse.get(hand,0.)+force*model.opt.timestep
                        support_force+=force
                if 'dressing_pad_contact' in geoms:
                    for digit in normals:
                        if any(b.startswith(f'g1_a/wuji_right/r_{digit}_') for b in bodies):
                            if first_pad_contact_time_s is None:first_pad_contact_time_s=float(data.time)
                            wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
                            force=max(0.,float(wrench[0]))
                            sign=1. if geoms[0]=='dressing_pad_contact' else -1.
                            normals[digit]+=sign*force*np.asarray(c.frame[:3])
                            if digit=='thumb':thumb_force+=force
                            else:index_force+=force
                if 'bandage_roll' in geoms or 'bandage_core' in geoms:
                    for digit in b_forces:
                        if any(b.startswith(f'g1_b/wuji_left/l_{digit}_') for b in bodies):
                            wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
                            force=max(0.,float(wrench[0]))
                            sign=1. if geoms[0] in ('bandage_roll','bandage_core') else -1.
                            b_forces[digit]+=force
                            b_normals[digit]+=sign*force*np.asarray(c.frame[:3])
            frame_peak=max(frame_peak,support_force)
            peak_support_force=max(peak_support_force,support_force)
            peak_thumb_force=max(peak_thumb_force,thumb_force)
            peak_index_force=max(peak_index_force,index_force)
            peak_b_thumb_force=max(peak_b_thumb_force,b_forces['thumb'])
            peak_b_index_force=max(peak_b_index_force,b_forces['index'])
            frame_foot &= supported=={'g1_a','g1_b'}
            foot_steps+=int(supported=={'g1_a','g1_b'})
            max_pad_displacement=max(max_pad_displacement,float(np.linalg.norm(data.xpos[pad_body]-pad_start)))
            max_spool_displacement=max(max_spool_displacement,float(np.linalg.norm(data.xpos[spool_body]-spool_start)))
        deep_steps+=int(frame_deep)
        valid=support_monitor.update(SupportSample(
            time_s=float(data.time),elbow_rad=float(data.qpos[elbow_qadr]),
            elbow_speed_rad_s=float(data.qvel[elbow_vadr]),peak_force_n=frame_peak,
            impulse_by_body_ns=frame_impulse,both_feet_supported=frame_foot,
            deep_nonfloor_contact=frame_deep))
        last_foot_supported=frame_foot
        last_left_contact=frame_peak>=.05
        last_right_contact=thumb_force>=.05 or index_force>=.05
        support_valid_steps+=int(valid)
        a,b=normals.values()
        cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>1e-8 else None
        opposed=bool(thumb_force>=.05 and index_force>=.05 and cos is not None and cos<-.5)
        opposed_run=opposed_run+1 if opposed else 0
        longest_opposed_run=max(longest_opposed_run,opposed_run)
        both_digits_opposed_steps+=int(opposed)
        ba,bb=b_normals.values()
        bcos=float(ba@bb/(np.linalg.norm(ba)*np.linalg.norm(bb))) if np.linalg.norm(ba)*np.linalg.norm(bb)>1e-8 else None
        b_opposed=bool(b_forces['thumb']>=.05 and b_forces['index']>=.05 and bcos is not None and bcos<-.5)
        b_opposed_run=b_opposed_run+1 if b_opposed else 0
        b_longest_opposed_run=max(b_longest_opposed_run,b_opposed_run)
        trace.append({'time_s':float(data.time),'segment':segment,'elbow_rad':float(data.qpos[elbow_qadr]),
                      'support_valid':valid,'support_force_n':frame_peak,'thumb_force_n':thumb_force,
                      'index_force_n':index_force,'normal_cosine':cos,'opposed':opposed,
                      'b_thumb_force_n':b_forces['thumb'],'b_index_force_n':b_forces['index'],
                      'b_normal_cosine':bcos,'b_opposed':b_opposed,
                      'pad_center_m':data.xpos[pad_body].tolist()})
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        break
report={'scope':('Dynamic double-G1 WBC screen: A-left forearm support, A-right pad pinch, B-left spool grasp attempt; no bandaging claim'
                 if args.spool_after else 'Dynamic double-G1 WBC screen: A-left support and A-right pinch; B stands; no bandaging claim'),
        'staged_pregrasp':args.staged,
        'spool_after':args.spool_after,'b_root_x_m':.55 if args.spool_after else .7,
        'settled_spool':args.settled_spool,
        'analytic_grip_specialist':args.analytic_grip_specialist,
        'b_analytic_grip':args.b_analytic_grip,
        'wbc_backend':args.wbc_backend,
        'hold_s':args.hold_s,
        'router_mode_counts':router_modes,
        'router_first_hand_specialist_time_s':first_specialist_time_s,
        'grasp_finger_weight':args.grasp_finger_weight,
        'b_finger_weight':args.b_finger_weight,
        'b_palm_position_weight':args.b_palm_position_weight,
        'a_right_palm_rotation_weight':args.palm_rotation_weight,
        'a_right_palm_position_weight':args.right_palm_position_weight,
        'simulated_time_s':float(data.time),'planned_duration_s':sum(durations),
        'wbc_hz':hz,'segments_completed':segment if not errors else segment-1,
        'forearm_mass_kg':.8,'wbc_forearm_force_cap_n':args.support_total_force_cap_n,
        'wbc_finger_force_cap_n':args.support_finger_force_cap_n,
        'final_elbow_rad':float(data.qpos[elbow_qadr]),
        'final_half_second_mean_abs_elbow_rad':float(np.mean([abs(t['elbow_rad']) for t in trace[-25:]])) if trace else None,
        'online_support_quality_final':support_monitor.last_report,
        'online_support_valid_fraction':support_valid_steps/max(1,len(trace)),
        'peak_a_left_forearm_force_n':peak_support_force,
        'foot_support_fraction':foot_steps/max(1,len(trace)*interval),
        'deep_nonfloor_robot_contact_control_steps':deep_steps,
        'worst_deep_contact':worst_deep_contact,'worst_penetration_m':worst_penetration_m,
        'first_pad_contact_time_s':first_pad_contact_time_s,
        'peak_thumb_force_n':peak_thumb_force,'peak_index_force_n':peak_index_force,
        'longest_opposed_contact_s':longest_opposed_run/hz,
        'opposed_contact_control_fraction':both_digits_opposed_steps/max(1,len(trace)),
        'max_pad_center_displacement_m':max_pad_displacement,
        'max_spool_center_displacement_m':max_spool_displacement,
        'peak_b_left_thumb_force_n':peak_b_thumb_force,
        'peak_b_left_index_force_n':peak_b_index_force,
        'longest_b_left_opposed_spool_contact_s':b_longest_opposed_run/hz,
        'final_pad_center_m':data.xpos[pad_body].tolist(),
        'final_thumb_tip_m':data.xpos[thumb_tip].tolist(),
        'final_index_tip_m':data.xpos[index_tip].tolist(),
        'final_right_arm_joint_error_max_rad':float(np.max(np.abs(data.qpos[right_arm_qadr]-path[-1,right_arm_qadr]))),
        'final_right_thumb_index_joint_error_max_rad':float(np.max(np.abs(data.qpos[right_digit_qadr]-path[-1,right_digit_qadr]))),
        'worst_right_digit_joints':sorted(({'joint':model.joint(j).name,
                                            'error_rad':float(abs(data.qpos[model.jnt_qposadr[j]]-
                                                                  path[-1,model.jnt_qposadr[j]]))}
                                           for j in range(model.njnt)
                                           if model.joint(j).name.startswith(('g1_a/wuji_right/r_thumb_',
                                                                               'g1_a/wuji_right/r_index_finger_'))),
                                          key=lambda row:-row['error_rad'])[:8],
        'finite_state':bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        'errors':errors,'trace':trace}
stem='joint_support_pinch_staged_wbc_report' if args.staged else 'joint_support_pinch_wbc_report'
if args.grasp_finger_weight!=1.:stem+=f'_fingerweight{str(args.grasp_finger_weight).replace(".","p")}'
if args.palm_rotation_weight!=10.:stem+=f'_rotation{str(args.palm_rotation_weight).replace(".","p")}'
if args.right_palm_position_weight!=1000.:stem+=f'_position{str(args.right_palm_position_weight).replace(".","p")}'
if args.analytic_grip_specialist:stem+='_specialist'
if args.hold_s!=1.5:stem+=f'_hold{str(args.hold_s).replace(".","p")}'
if args.support_finger_force_cap_n!=2.:stem+=f'_fingercap{str(args.support_finger_force_cap_n).replace(".","p")}'
if args.support_total_force_cap_n!=12.:stem+=f'_totalcap{str(args.support_total_force_cap_n).replace(".","p")}'
if args.spool_after:stem+='_spool'
if args.settled_spool:stem+='_settled'
if args.b_analytic_grip:stem+='_bfeedback'
if args.b_finger_weight!=1.:stem+='_bfinger'+str(args.b_finger_weight).replace('.','p')
if args.b_palm_position_weight!=1000.:stem+='_bposition'+str(args.b_palm_position_weight).replace('.','p')
if args.wbc_backend!='CLARABEL':stem+='_'+args.wbc_backend.lower()
(OUT/f'{stem}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
np.savez_compressed(OUT/f'{stem}_final_state.npz',qpos=data.qpos,qvel=data.qvel,reference_qpos=path[-1])
print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
