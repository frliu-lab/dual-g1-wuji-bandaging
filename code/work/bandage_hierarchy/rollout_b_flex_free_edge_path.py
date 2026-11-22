"""Closed-loop dual-G1 WBC test of the collision-screened free-edge approach."""
import json
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC
from flex_contact_jac import flex_contact_jacobian

OUT = ROOT/'outputs/wuji_multiagent/bandage_v0'
parser = argparse.ArgumentParser()
parser.add_argument('--track-edge', action='store_true')
parser.add_argument('--edge-damping', type=float, default=.2)
parser.add_argument('--pinchtab',action='store_true')
parser.add_argument('--start-at-goal',action='store_true')
parser.add_argument('--duration-s',type=float,default=1.)
parser.add_argument('--flex-tip-force-n',type=float,default=None)
parser.add_argument('--wrist-weight',type=float,default=None)
parser.add_argument('--local-physics-correction',action='store_true')
parser.add_argument('--stabilize-other-digits',action='store_true')
parser.add_argument('--material-tip-follow',action='store_true')
parser.add_argument('--tip-closure-mm',type=float,default=0.)
parser.add_argument('--tip-weight',type=float,default=1000.)
args = parser.parse_args()
suffix = f'_damping{args.edge_damping:g}' if args.edge_damping != .2 else ''
SCENE = OUT/(f'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab_bulged.xml'
             if args.pinchtab else
             f'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5{suffix}.xml')
PATH = OUT/('joint_b_pinchtab_rrt_path.npz' if args.pinchtab else
            'joint_b_flex_free_edge_rrt_path.npz')
m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)
path_file = np.load(PATH)
waypoints = path_file['qpos']
joint_names = [str(s) for s in path_file['joint_names']]
qa = np.array([m.joint(s).qposadr[0] for s in joint_names])
va = np.array([m.joint(s).dofadr[0] for s in joint_names])
wrist_joint=m.joint('g1_b/right_wrist_roll_joint').id
wrist_qadr=int(m.jnt_qposadr[wrist_joint]);wrist_vadr=int(m.jnt_dofadr[wrist_joint])
wrist_actuator=int(np.flatnonzero(m.actuator_trnid[:,0]==wrist_joint)[0])
d.qpos[:] = waypoints[-1 if args.start_at_goal else 0]
mujoco.mj_forward(m, d)

palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
def relevant_contact(model, c):
    return any(f >= 0 for f in c.flex) or any(
        model.body(model.geom_bodyid[g]).name.startswith(('g1_a/', 'g1_b/'))
        for g in c.geom if g >= 0)

wbc = JointWBC(m, d.qpos.copy(), palm_names=palms, contact_filter=relevant_contact)
wbc.backend = 'DIRECT'
wbc.flex_tip_normal_goal_n=args.flex_tip_force_n
if args.stabilize_other_digits:
    wbc.hand_position_mode=True
    for i in wbc.hand_indices:
        wbc.hand_commands[i]=d.qpos[wbc.act_qadr[i]]
    wbc.force_control_indices={int(i) for i in wbc.hand_indices
                               if m.actuator(int(i)).name.startswith('g1_b/wuji_right/r_') and
                               any(token in m.actuator(int(i)).name for token in ('THJ','FFJ'))}
if args.wrist_weight is not None:
    wbc.weights[wrist_vadr]=args.wrist_weight
interval = round(.02/m.opt.timestep)
duration = args.duration_s
initial_edge = d.flexvert_xpos[64:66].mean(axis=0).copy()
initial_row28 = d.flexvert_xpos[56:58].mean(axis=0).copy()
screen = json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
tip_names = ('g1_b/wuji_right/r_thumb_tip_sensor_frame',
             'g1_b/wuji_right/r_index_finger_tip_sensor_frame')
tip_target = np.array(screen['results'][0]['target_m'])
if args.track_edge:
    wbc.tip_position_weight = 1000.
material_targets={}
if args.material_tip_follow:
    if not args.start_at_goal:
        parser.error('Material-point fingertip following currently requires --start-at-goal')
    wbc.tip_position_weight=args.tip_weight
    for digit,tip_name in [('thumb',tip_names[0]),('index',tip_names[1])]:
        contacts=[]
        for c in d.contact:
            if not any(f>=0 for f in c.flex) or c.dist>=.003:continue
            if not any(m.body(m.geom_bodyid[int(g)]).name==tip_name for g in c.geom if g>=0):continue
            side=0 if c.flex[0]>=0 else 1
            _,_,verts,weights=flex_contact_jacobian(m,d,c,side)
            point=weights@d.flexvert_xpos[verts]
            contacts.append((float(c.dist),verts,weights,point))
        if not contacts:
            raise RuntimeError(f'No initial {digit} flex contact for material target')
        _,verts,weights,point=min(contacts,key=lambda x:x[0])
        offset=d.xpos[m.body(tip_name).id]-point
        offset*=max(0.,1.-args.tip_closure_mm/1000/max(np.linalg.norm(offset),1e-9))
        material_targets[tip_name]=(verts,weights,offset)
max_pen = 0.0
max_tip_error = 0.0
feet_count = 0
flex_contact_count = 0
opposed_contact_count = 0
edge_drift = 0.0
solves = 0
error = None
trace = []

def ref_at(u):
    scaled = np.clip(u, 0, 1)*(len(waypoints)-1)
    i = min(len(waypoints)-2, int(scaled))
    t = scaled-i
    return waypoints[i]*(1-t)+waypoints[i+1]*t

try:
    for k in range(round(duration/.02)):
        progress = (k+1)/round(duration/.02)
        smooth = 1. if args.start_at_goal else progress*progress*(3-2*progress)
        wbc.qref[:] = ref_at(smooth)
        if args.track_edge:
            wbc.tip_position_targets = {
                tip_names[0]: tip_target[0] + d.flexvert_xpos[64:66].mean(axis=0)-initial_edge,
                tip_names[1]: tip_target[1] + d.flexvert_xpos[56:58].mean(axis=0)-initial_row28,
            }
        if args.material_tip_follow:
            wbc.tip_position_targets={name:weights@d.flexvert_xpos[verts]+offset
                                      for name,(verts,weights,offset) in material_targets.items()}
        ctrl, meta, _ = wbc.solve(d)
        d.ctrl[:] = ctrl
        mujoco.mj_forward(m,d)
        pre_step_wrist_actual_acceleration=float(d.qacc[wrist_vadr])
        pre_step_wrist_actual_constraint_force=float(d.qfrc_constraint[wrist_vadr])
        wrist_control_correction=0.
        if args.local_physics_correction:
            base_ctrl=float(d.ctrl[wrist_actuator])
            d.ctrl[wrist_actuator]=base_ctrl+.1
            mujoco.mj_forward(m,d)
            sensitivity=(float(d.qacc[wrist_vadr])-pre_step_wrist_actual_acceleration)/.1
            if np.isfinite(sensitivity) and sensitivity>1.:
                wrist_control_correction=float(np.clip(
                    (wbc.last_predicted_acceleration[wrist_vadr]-pre_step_wrist_actual_acceleration)/sensitivity,
                    -2.,2.))
            low,high=m.actuator_ctrlrange[wrist_actuator] if m.actuator_ctrllimited[wrist_actuator] else (-1e4,1e4)
            d.ctrl[wrist_actuator]=np.clip(base_ctrl+wrist_control_correction,low,high)
            wrist_control_correction=float(d.ctrl[wrist_actuator]-base_ctrl)
            mujoco.mj_forward(m,d)
        solves += 1
        interval_opposed = 0
        interval_max_pen = 0.
        first_step_actual_wrist_acceleration=None
        for substep in range(interval):
            mujoco.mj_step(m, d)
            if substep==0:
                first_step_actual_wrist_acceleration=float(d.qacc[wrist_vadr])
            if not np.all(np.isfinite(d.qpos)):
                raise RuntimeError('Nonfinite state')
            feet = {'a': set(), 'b': set()}
            has_forearm_flex = False
            tip_normals={'thumb':[],'index':[]}
            for c in d.contact:
                geoms = [int(g) for g in c.geom if g >= 0]
                names = [m.geom(g).name for g in geoms]
                bodies = [m.body(m.geom_bodyid[g]).name for g in geoms]
                flex = any(f >= 0 for f in c.flex)
                if flex and 'training_forearm_dynamic' in bodies:
                    has_forearm_flex = True
                if flex and c.dist<.003:
                    if any('g1_b/wuji_right/r_thumb_tip_sensor_frame'==n for n in bodies):
                        tip_normals['thumb'].append(c.frame[:3].copy())
                    if any('g1_b/wuji_right/r_index_finger_tip_sensor_frame'==n for n in bodies):
                        tip_normals['index'].append(c.frame[:3].copy())
                if c.dist < 0 and 'floor' not in names and any(n.startswith('g1_b/') for n in bodies):
                    max_pen = max(max_pen, -float(c.dist))
                    interval_max_pen=max(interval_max_pen,-float(c.dist))
                if 'floor' in names:
                    for agent in ('a', 'b'):
                        for side in ('left', 'right'):
                            if any(n.startswith(f'g1_{agent}/{side}_official_foot_contact_') for n in names):
                                feet[agent].add(side)
            feet_count += all(feet[a] == {'left', 'right'} for a in ('a', 'b'))
            flex_contact_count += has_forearm_flex
            opposed_here=any(np.dot(a,b)<-.5 for a in tip_normals['thumb'] for b in tip_normals['index'])
            opposed_contact_count += opposed_here
            interval_opposed += opposed_here
            edge_drift = max(edge_drift, float(np.linalg.norm(d.flexvert_xpos[64:66].mean(axis=0)-initial_edge)))
        tip_error = float(np.max(np.abs(d.qpos[qa]-wbc.qref[qa])))
        max_tip_error = max(max_tip_error, tip_error)
        trace.append({'time_s': float(d.time), 'path_progress': float(smooth),
                      'max_joint_tracking_error_rad': tip_error,
                      'worst_joint_name':joint_names[int(np.argmax(np.abs(d.qpos[qa]-wbc.qref[qa])))],
                      'opposed_tip_contact_fraction':interval_opposed/interval,
                      'max_b_nonfloor_penetration_m':interval_max_pen,
                      'b_right_wrist_roll_rad':float(d.qpos[wrist_qadr]),
                      'b_right_wrist_roll_ref_rad':float(wbc.qref[wrist_qadr]),
                      'b_right_wrist_roll_velocity_rad_s':float(d.qvel[wrist_vadr]),
                      'b_right_wrist_roll_requested_torque_Nm':float(wbc.last_tau[wrist_actuator]),
                      'b_right_wrist_roll_predicted_acceleration_rad_s2':float(wbc.last_predicted_acceleration[wrist_vadr]),
                      'b_right_wrist_roll_pre_step_actual_acceleration_rad_s2':pre_step_wrist_actual_acceleration,
                      'b_right_wrist_roll_pre_step_actual_constraint_force_Nm':pre_step_wrist_actual_constraint_force,
                      'b_right_wrist_roll_predicted_contact_generalized_force_Nm':float(wbc.last_predicted_contact_generalized_force[wrist_vadr]),
                      'b_right_wrist_roll_first_step_actual_acceleration_rad_s2':first_step_actual_wrist_acceleration,
                      'b_right_wrist_roll_actual_actuator_force_Nm':float(d.actuator_force[wrist_actuator]),
                      'b_right_wrist_roll_local_control_correction':wrist_control_correction,
                      'b_right_wrist_roll_constraint_force_Nm':float(d.qfrc_constraint[wrist_vadr]),
                      'b_right_wrist_roll_bias_force_Nm':float(d.qfrc_bias[wrist_vadr]),
                      'b_right_wrist_roll_passive_force_Nm':float(d.qfrc_passive[wrist_vadr]),
                      'free_edge_drift_m': float(np.linalg.norm(d.flexvert_xpos[64:66].mean(axis=0)-initial_edge)),
                      'contact_task_residual_m_s2': meta['contact_acceleration_task_residual_m_s2']})
except Exception as exc:
    error = f'{type(exc).__name__}: {exc}'

steps = max(1, solves*interval)
report = {'scope': 'Dynamic WBC tracking of a static path toward a flexible free tab; no force-closed grasp or completed wrap',
          'track_edge_tip_targets': args.track_edge,
          'pinchtab':args.pinchtab,
          'start_at_goal':args.start_at_goal,
          'edge_damping':args.edge_damping,
          'flex_tip_force_goal_n':args.flex_tip_force_n,
          'b_right_wrist_roll_acceleration_weight':args.wrist_weight,
          'local_physics_correction':args.local_physics_correction,
          'stabilize_other_digits':args.stabilize_other_digits,
          'material_tip_follow':args.material_tip_follow,
          'tip_closure_mm':args.tip_closure_mm,
          'tip_position_weight':args.tip_weight if args.material_tip_follow else None,
          'duration_s': float(d.time), 'planned_duration_s': duration, 'control_solves': solves,
          'error': error, 'two_robot_two_foot_fraction': feet_count/steps,
          'flex_forearm_contact_fraction': flex_contact_count/steps,
          'opposed_b_tip_flex_contact_fraction':opposed_contact_count/steps,
          'max_b_nonfloor_penetration_m': max_pen,
          'max_free_edge_drift_m': edge_drift,
          'max_joint_tracking_error_rad': max_tip_error,
          'final_joint_target_error_rad': float(np.max(np.abs(d.qpos[qa]-waypoints[-1, qa]))),
          'trace': trace}
report_name = (('joint_b_pinchtab_dynamic_path_rollout' if args.pinchtab else
                'joint_b_flex_free_edge_dynamic_path_rollout')+
               ('_goal_hold' if args.start_at_goal else '')+
               ('_track_edge' if args.track_edge else '')+
               (f'_flexforce{args.flex_tip_force_n:g}' if args.flex_tip_force_n is not None else '')+
               (f'_wristweight{args.wrist_weight:g}' if args.wrist_weight is not None else '')+
               ('_localphysics' if args.local_physics_correction else '')+
               ('_stabilizedigits' if args.stabilize_other_digits else '')+
               ('_materialfollow' if args.material_tip_follow else '')+
               (f'_closure{args.tip_closure_mm:g}mm' if args.tip_closure_mm else '')+
               (f'_tipweight{args.tip_weight:g}' if args.material_tip_follow and args.tip_weight!=1000. else '')+
               suffix+'.json')
(OUT/report_name).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
