"""Compare a moving forearm with and without the provisional A-left contact path."""
import json
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--candidate-path',action='store_true')
parser.add_argument('--candidate-root-x',type=float,default=-.3)
parser.add_argument('--candidate-elbow-angle',type=float,default=.25)
parser.add_argument('--reach-fraction',type=float,default=1.)
parser.add_argument('--segment-s',type=float,default=None)
parser.add_argument('--index-extra',type=float,default=0.)
parser.add_argument('--forearm-follow-gain',type=float,default=0.)
parser.add_argument('--follow-start-fraction',type=float,default=1.)
parser.add_argument('--force-feedback-gain',type=float,default=0.)
parser.add_argument('--force-target-n',type=float,default=8.)
parser.add_argument('--forearm-solref-time',type=float,default=None)
parser.add_argument('--forearm-mass-kg',type=float,default=.8)
parser.add_argument('--armrest',action='store_true')
parser.add_argument('--armrest-top-z',type=float,default=.795)
parser.add_argument('--armrest-profile',choices=('side','proximal'),default='side')
parser.add_argument('--easing',choices=('cubic','frontloaded'),default='cubic')
args=parser.parse_args()
sys.path.insert(0, str(ROOT / 'work/wuji_multiagent'))
from joint_wbc import JointWBC
from support_quality import SupportQualityMonitor, SupportSample

old = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
path_old = np.load(OUT / 'forearm_support_rrt.npz')['qpos']
scene = ET.parse(OUT / 'dynamic_forearm_variant.xml').getroot()
scene.find(".//body[@name='g1_a/pelvis']").set('pos', f'{args.candidate_root_x} 0 .8' if args.candidate_path else '-.58 0 .8')
scene.find(".//geom[@name='manikin_forearm']").set('mass',str(args.forearm_mass_kg))
if args.armrest:
    pad_pos=('.04 -.365' if args.armrest_profile=='side' else '-.08 -.34')
    pad_size=('.10 .015 .035' if args.armrest_profile=='side' else '.05 .04 .035')
    ET.SubElement(scene.find(".//body[@name='training_manikin']"),'geom',{
        'name':'manikin_armrest_pad','type':'box','pos':f'{pad_pos} {args.armrest_top_z-.035}',
        'size':pad_size,'rgba':'.35 .55 .65 1','friction':'.8 .005 .0001'})
if args.forearm_solref_time is not None:
    scene.find(".//geom[@name='manikin_forearm']").set('solref',f'{args.forearm_solref_time} 1')
model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))

def remap(q):
    result = model.qpos0.copy()
    for j in range(old.njnt):
        name = old.joint(j).name
        new_id = model.joint(name).id
        n = 7 if old.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else (4 if old.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1)
        result[model.jnt_qposadr[new_id]:model.jnt_qposadr[new_id]+n] = q[old.jnt_qposadr[j]:old.jnt_qposadr[j]+n]
    return result

path_suffix=str(args.candidate_root_x).replace('-','m').replace('.','p')
if args.candidate_elbow_angle!=.25:path_suffix+='_elbow'+str(args.candidate_elbow_angle).replace('.','p')
path_file=(OUT/'dynamic_palm_support_rrt.npz' if args.candidate_root_x==-.3 and args.candidate_elbow_angle==.25 else
           OUT/('dynamic_palm_support_rrt_root_'+path_suffix+'.npz'))
path = (np.load(path_file)['qpos'] if args.candidate_path
        else np.array([remap(q) for q in path_old]))
if args.candidate_path:
    path[-1]=path[0]+args.reach_fraction*(path[-1]-path[0])
    if args.index_extra:
        for j in range(model.njnt):
            name=model.joint(j).name
            if not name.startswith('g1_a/wuji_left/l_index_') or 'abd' in name:
                continue
            if not any(name.endswith(s) for s in ('_mcp_flex','_pip','_dip')):
                continue
            a=int(model.jnt_qposadr[j]);lo,hi=model.jnt_range[j]
            path[:,a]=np.clip(path[:,a]+args.index_extra,lo+.03,hi-.03)
palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
def contact_filter(m, c):
    return any(m.body(m.geom_bodyid[int(g)]).name.startswith(('g1_a/', 'g1_b/'))
               for g in c.geom if g >= 0)

jid = model.joint('training_elbow_hinge').id
qadr = int(model.jnt_qposadr[jid])
vadr = int(model.jnt_dofadr[jid])
geom_id = model.geom('manikin_forearm').id
forearm_body=model.body('training_forearm_dynamic').id
palm_body=model.body('g1_a/wuji_left/l_wrist').id
arm_joints=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_a/left_')
            and any(s in model.joint(j).name for s in ('shoulder','elbow','wrist'))
            and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
arm_qadr=np.array([model.jnt_qposadr[j] for j in arm_joints],dtype=int)
arm_vadr=np.array([model.jnt_dofadr[j] for j in arm_joints],dtype=int)
goal_data=mujoco.MjData(model);goal_data.qpos[:]=path[-1];mujoco.mj_forward(model,goal_data)
goal_fpos=goal_data.xpos[forearm_body].copy()
goal_frot=goal_data.xmat[forearm_body].reshape(3,3).copy()
goal_ppos=goal_data.xpos[palm_body].copy()
goal_prot=goal_data.xmat[palm_body].reshape(3,3).copy()
relative_palm_pos=goal_frot.T@(goal_ppos-goal_fpos)
relative_palm_rot=goal_frot.T@goal_prot
ref_data=mujoco.MjData(model)
def follow_forearm(qref,live,gain):
    """One bounded differential IK update on robot joints only."""
    qref[qadr]=live.qpos[qadr]
    ref_data.qpos[:]=qref;mujoco.mj_forward(model,ref_data)
    frot=live.xmat[forearm_body].reshape(3,3)
    current_rot=ref_data.xmat[palm_body].reshape(3,3)
    forearm_displacement=(live.xpos[forearm_body]+frot@relative_palm_pos)-goal_ppos
    target_pos=ref_data.xpos[palm_body]+forearm_displacement
    target_rot=(frot@goal_frot.T)@current_rot
    pos_error=target_pos-ref_data.xpos[palm_body]
    rot_error=.5*sum((np.cross(current_rot[:,a],target_rot[:,a]) for a in range(3)),start=np.zeros(3))
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
    mujoco.mj_jacBody(model,ref_data,jp,jr,palm_body)
    J=np.vstack((jp[:,arm_vadr],jr[:,arm_vadr]))
    delta=J.T@np.linalg.solve(J@J.T+.02**2*np.eye(6),np.r_[pos_error,rot_error])
    qref[arm_qadr]+=gain*np.clip(delta,-.12,.12)
    for j,a in zip(arm_joints,arm_qadr):
        lo,hi=model.jnt_range[j];qref[a]=np.clip(qref[a],lo+.03,hi-.03)
    return float(np.linalg.norm(pos_error))
def vertical_palm_offset(qref, offset):
    """Apply a bounded world-z palm correction to the nominal arm reference."""
    ref_data.qpos[:]=qref;mujoco.mj_forward(model,ref_data)
    jp=np.zeros((3,model.nv))
    mujoco.mj_jacBody(model,ref_data,jp,None,palm_body)
    J=jp[:,arm_vadr]
    delta=J.T@np.linalg.solve(J@J.T+.02**2*np.eye(3),np.array([0.,0.,offset]))
    qref[arm_qadr]+=np.clip(delta,-.12,.12)
    for j,a in zip(arm_joints,arm_qadr):
        lo,hi=model.jnt_range[j];qref[a]=np.clip(qref[a],lo+.03,hi-.03)
def measured_forearm_force(live):
    value=0.
    for ci,c in enumerate(live.contact):
        if c.geom1<0 or c.geom2<0 or geom_id not in (c.geom1,c.geom2):continue
        if not any(model.body(model.geom_bodyid[g]).name.startswith('g1_a/wuji_left/') for g in (c.geom1,c.geom2)):
            continue
        wrench=np.zeros(6);mujoco.mj_contactForce(model,live,ci,wrench)
        value+=max(0.,float(wrench[0]))
    return value
hz = 50
interval = round(1 / hz / model.opt.timestep)
duration = 3.8
segment_s=args.segment_s if args.segment_s is not None else (1.2 if args.candidate_path else .6)
out = {}
for condition in ('no_contact', 'palm_candidate_path' if args.candidate_path else 'provisional_pinky_path'):
    data = mujoco.MjData(model)
    data.qpos[:] = path[0]
    mujoco.mj_forward(model, data)
    wbc = JointWBC(model, data.qpos.copy(), palm_names=palms, contact_filter=contact_filter)
    trace = []
    errors = []
    impulse_by_body = {}
    last_half_impulse_by_body={}
    foot_support = 0
    deep = 0
    peak_force = 0.
    peak_force_time=0.
    max_follow_error=0.
    force_offset_z=0.
    contact_engaged=False
    support_monitor=SupportQualityMonitor()
    online_valid_steps=0
    first_online_valid_time=None
    initial = float(data.qpos[qadr])
    for k in range(round(duration * hz)):
        if condition != 'no_contact':
            segment_steps=round(segment_s*hz)
            segment = min(k // segment_steps + 1, len(path) - 1)
            u = min(1., (k % segment_steps + 1) / segment_steps) if k < (len(path)-1)*segment_steps else 1.
            u = u*u*(3-2*u) if args.easing=='cubic' else (1-(1-u)**4)**2
            wbc.qref[:] = (1-u)*path[segment-1] + u*path[segment]
            if args.candidate_path and args.forearm_follow_gain:
                progress=min(1.,k/max(1,round(segment_s*hz)*(len(path)-1)))
                if progress>=args.follow_start_fraction:
                    ramp=min(1.,(progress-args.follow_start_fraction)/max(.001,1.-args.follow_start_fraction))
                    effective_gain=args.forearm_follow_gain*(ramp if args.follow_start_fraction<1. else 1.)
                    max_follow_error=max(max_follow_error,follow_forearm(wbc.qref,data,effective_gain))
            if args.candidate_path and args.force_feedback_gain:
                measured=measured_forearm_force(data)
                contact_engaged=contact_engaged or measured>=.05
                if contact_engaged:
                    force_offset_z=float(np.clip(force_offset_z+args.force_feedback_gain*(args.force_target_n-measured)/hz,-.025,.005))
                    vertical_palm_offset(wbc.qref,force_offset_z)
        else:
            wbc.qref[:] = path[0]
        try:
            data.ctrl[:], _, _ = wbc.solve(data)
            frame_impulse={}
            frame_peak=0.
            frame_both_feet=True
            frame_deep=False
            for _ in range(interval):
                mujoco.mj_step(model, data)
                supported = set()
                force = 0.
                for ci, c in enumerate(data.contact):
                    if c.geom1 < 0 or c.geom2 < 0:
                        continue
                    names = [model.geom(g).name for g in (c.geom1, c.geom2)]
                    bodies = [model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
                    if 'floor' in names:
                        for label in ('g1_a', 'g1_b'):
                            if any(b.startswith(label + '/') for b in bodies):
                                supported.add(label)
                    elif any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies):
                        if c.dist < -.003:
                            deep += 1
                            frame_deep=True
                        if 'manikin_forearm' in names:
                            hand = next((b for b in bodies if b.startswith('g1_a/wuji_left/')), None)
                            if hand:
                                wrench = np.zeros(6)
                                mujoco.mj_contactForce(model, data, ci, wrench)
                                fn = max(0., float(wrench[0]))
                                force += fn
                                impulse_by_body[hand] = impulse_by_body.get(hand, 0.) + fn*model.opt.timestep
                                frame_impulse[hand]=frame_impulse.get(hand,0.)+fn*model.opt.timestep
                                if data.time>=duration-.5:
                                    last_half_impulse_by_body[hand]=last_half_impulse_by_body.get(hand,0.)+fn*model.opt.timestep
                foot_support += int(supported == {'g1_a', 'g1_b'})
                frame_both_feet=frame_both_feet and supported=={'g1_a','g1_b'}
                frame_peak=max(frame_peak,force)
                if force>peak_force:peak_force=force;peak_force_time=float(data.time)
            online_valid=support_monitor.update(SupportSample(
                time_s=float(data.time),elbow_rad=float(data.qpos[qadr]),
                elbow_speed_rad_s=float(data.qvel[vadr]),peak_force_n=frame_peak,
                impulse_by_body_ns=frame_impulse,both_feet_supported=frame_both_feet,
                deep_nonfloor_contact=frame_deep))
            online_valid_steps+=int(online_valid)
            if online_valid and first_online_valid_time is None:first_online_valid_time=float(data.time)
            trace.append({'time_s': float(data.time), 'elbow_rad': float(data.qpos[qadr]),
                          'elbow_speed_rad_s': float(data.qvel[vadr]),
                          'forearm_center_m': data.geom_xpos[geom_id].tolist(),
                          'a_left_forearm_force_n': force,
                          'online_support_valid':online_valid})
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
            break
    final = trace[-25:]
    if not final:
        raise RuntimeError(f'{condition} produced no valid WBC samples: {errors}')
    out[condition] = {'simulated_time_s': float(data.time), 'initial_elbow_rad': initial,
        'final_elbow_rad': float(data.qpos[qadr]),
        'final_half_second_mean_abs_elbow_rad': float(np.mean([abs(x['elbow_rad']) for x in final])),
        'final_half_second_mean_force_n': float(np.mean([x['a_left_forearm_force_n'] for x in final])),
        'final_half_second_contact_fraction': float(np.mean([x['a_left_forearm_force_n'] >= .05 for x in final])),
        'peak_a_left_forearm_force_n': peak_force,
        'peak_force_time_s':peak_force_time,
        'max_forearm_follow_reference_error_m':max_follow_error,
        'final_force_feedback_offset_z_m':force_offset_z,
        'online_support_quality_final':support_monitor.last_report,
        'online_support_valid_fraction':online_valid_steps/max(1,len(trace)),
        'first_online_support_valid_time_s':first_online_valid_time,
        'a_left_impulse_by_hand_body_ns': impulse_by_body,
        'final_half_second_impulse_by_hand_body_ns':last_half_impulse_by_body,
        'foot_support_fraction': foot_support/max(1, len(trace)*interval),
        'deep_nonfloor_robot_contact_count': deep,
        'finite_state': bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        'errors': errors, 'trace': trace}
report = {'scope': ('Dynamic training-forearm WBC counterfactual for a static palm candidate; no validity claim' if args.candidate_path else
                    'Dynamic training-forearm WBC counterfactual; provisional path is pinky-only and rejected as humanlike support'),
          'forearm_mass_kg':args.forearm_mass_kg, 'candidate_root_x_m':args.candidate_root_x if args.candidate_path else None,'candidate_initial_elbow_rad':args.candidate_elbow_angle if args.candidate_path else None,'armrest_variant':args.armrest,'armrest_profile':args.armrest_profile if args.armrest else None,'armrest_top_z_m':args.armrest_top_z if args.armrest else None,'forearm_solref_time_s':args.forearm_solref_time,'reach_fraction':args.reach_fraction,'segment_s':segment_s,'easing':args.easing,'index_extra_rad':args.index_extra,'forearm_follow_gain':args.forearm_follow_gain,'follow_start_fraction':args.follow_start_fraction,'force_feedback_gain':args.force_feedback_gain,'force_target_n':args.force_target_n,'conditions': out}
name='dynamic_forearm_support_comparison.json'
if args.candidate_path:
    name='dynamic_palm_support_wbc_comparison'
    if args.reach_fraction!=1.:name+='_reach'+str(args.reach_fraction).replace('.','p')
    if args.segment_s is not None:name+='_segment'+str(segment_s).replace('.','p')
    if args.index_extra:name+='_index'+str(args.index_extra).replace('.','p')
    if args.forearm_follow_gain:name+='_follow'+str(args.forearm_follow_gain).replace('.','p')
    if args.follow_start_fraction!=1.:name+='_start'+str(args.follow_start_fraction).replace('.','p')
    if args.force_feedback_gain:name+='_force'+str(args.force_feedback_gain).replace('.','p')
    if args.forearm_solref_time is not None:name+='_solref'+str(args.forearm_solref_time).replace('.','p')
    if args.forearm_mass_kg!=.8:name+='_mass'+str(args.forearm_mass_kg).replace('.','p')
    if args.armrest:name+='_armrest'+args.armrest_profile+str(args.armrest_top_z).replace('.','p')
    if args.easing!='cubic':name+='_easing'+args.easing
    if args.candidate_root_x!=-.3:name+='_root'+str(args.candidate_root_x).replace('-','m').replace('.','p')
    if args.candidate_elbow_angle!=.25:name+='_elbow'+str(args.candidate_elbow_angle).replace('.','p')
    name+='.json'
(OUT/name).write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps({k: {j:v for j,v in s.items() if j!='trace'} for k,s in out.items()}, indent=2))
