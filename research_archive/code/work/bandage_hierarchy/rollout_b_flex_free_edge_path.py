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
parser.add_argument('--extended-tab',action='store_true')
parser.add_argument('--start-at-goal',action='store_true')
parser.add_argument('--duration-s',type=float,default=1.)
parser.add_argument('--flex-tip-force-n',type=float,default=None)
parser.add_argument('--measured-tip-force-n',type=float,default=None)
parser.add_argument('--pinch-bias-preset',choices=('balanced_a','balanced_b'),default=None)
parser.add_argument('--pinch-bias-until-s',type=float,default=None)
parser.add_argument('--flex-friction',type=float,default=None)
parser.add_argument('--wrist-weight',type=float,default=None)
parser.add_argument('--local-physics-correction',action='store_true')
parser.add_argument('--stabilize-other-digits',action='store_true')
parser.add_argument('--stabilize-all-digits',action='store_true')
parser.add_argument('--material-tip-follow',action='store_true')
parser.add_argument('--tip-closure-mm',type=float,default=0.)
parser.add_argument('--tip-weight',type=float,default=1000.)
parser.add_argument('--flex-measured-force-prior-weight',type=float,default=0.)
parser.add_argument('--releasable-fixture',action='store_true')
parser.add_argument('--fixture-release-s',type=float,default=None)
parser.add_argument('--pinch-closure-mm',type=float,default=0.)
parser.add_argument('--b-right-palm-weight',type=float,default=1000.)
args = parser.parse_args()
if args.extended_tab and (not args.pinchtab or not args.start_at_goal):
    parser.error('--extended-tab requires --pinchtab --start-at-goal; the existing path has a different flex state')
if args.fixture_release_s is not None and not args.releasable_fixture:
    parser.error('--fixture-release-s requires --releasable-fixture')
if args.measured_tip_force_n is not None and (args.measured_tip_force_n<=0 or not args.pinchtab):
    parser.error('--measured-tip-force-n requires a positive force and --pinchtab')
if args.flex_friction is not None and (not np.isfinite(args.flex_friction) or args.flex_friction<=0):
    parser.error('--flex-friction must be positive and finite')
if args.pinch_bias_until_s is not None and (args.pinch_bias_preset is None or args.pinch_bias_until_s<0):
    parser.error('--pinch-bias-until-s requires a preset and nonnegative time')
suffix = f'_damping{args.edge_damping:g}' if args.edge_damping != .2 else ''
SCENE = OUT/(('joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml' if args.releasable_fixture else 'joint_dual_g1_pinchtab_extended20mm.xml') if args.extended_tab else
             'joint_dual_g1_pinchtab_releasable_fixture.xml' if args.releasable_fixture else
             f'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab_bulged.xml'
             if args.pinchtab else
             f'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5{suffix}.xml')
PATH = OUT/('joint_b_pinchtab_rrt_path.npz' if args.pinchtab else
            'joint_b_flex_free_edge_rrt_path.npz')
m = mujoco.MjModel.from_xml_path(str(SCENE))
if args.flex_friction is not None:
    m.flex_friction[:,0]=args.flex_friction
d = mujoco.MjData(m)
path_file = np.load(PATH)
waypoints = path_file['qpos']
joint_names = [str(s) for s in path_file['joint_names']]
qa = np.array([m.joint(s).qposadr[0] for s in joint_names])
va = np.array([m.joint(s).dofadr[0] for s in joint_names])
wrist_joint=m.joint('g1_b/right_wrist_roll_joint').id
wrist_qadr=int(m.jnt_qposadr[wrist_joint]);wrist_vadr=int(m.jnt_dofadr[wrist_joint])
wrist_actuator=int(np.flatnonzero(m.actuator_trnid[:,0]==wrist_joint)[0])
if args.extended_tab:
    d.qpos[:167] = waypoints[-1,:167]
else:
    d.qpos[:] = waypoints[-1 if args.start_at_goal else 0]
mujoco.mj_forward(m, d)
tip_names = ('g1_b/wuji_right/r_thumb_tip_sensor_frame',
             'g1_b/wuji_right/r_index_finger_tip_sensor_frame')
tip_ids=[m.body(name).id for name in tip_names]
pinch_actuators=[i for i in range(m.nu) if m.actuator(i).name.startswith('g1_b/wuji_right/r_') and
                 any(token in m.actuator(i).name for token in ('THJ','FFJ'))]
thumb_tip_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==tip_ids[0]]
index_tip_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==tip_ids[1]]
initial_tip_positions=np.array([d.xpos[bid].copy() for bid in tip_ids])

palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
def relevant_contact(model, c):
    return any(f >= 0 for f in c.flex) or any(
        model.body(model.geom_bodyid[g]).name.startswith(('g1_a/', 'g1_b/'))
        for g in c.geom if g >= 0)

wbc = JointWBC(m, d.qpos.copy(), palm_names=palms, contact_filter=relevant_contact,
               palm_position_weights=[1000.,1000.,1000.,args.b_right_palm_weight])
wbc.backend = 'DIRECT'
wbc.flex_tip_normal_goal_n=args.flex_tip_force_n
wbc.flex_measured_force_prior_weight=args.flex_measured_force_prior_weight
if args.stabilize_other_digits or args.stabilize_all_digits:
    wbc.hand_position_mode=True
    for i in wbc.hand_indices:
        wbc.hand_commands[i]=d.qpos[wbc.act_qadr[i]]
    if args.stabilize_other_digits:
        wbc.force_control_indices={int(i) for i in wbc.hand_indices
                                   if m.actuator(int(i)).name.startswith('g1_b/wuji_right/r_') and
                                   any(token in m.actuator(int(i)).name for token in ('THJ','FFJ'))}
    if args.pinch_closure_mm:
        if not args.stabilize_all_digits:
            parser.error('--pinch-closure-mm requires --stabilize-all-digits')
        thumb_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==m.body(tip_names[0]).id]
        index_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==m.body(tip_names[1]).id]
        pinch_actuators=[int(i) for i in wbc.hand_indices
                         if m.actuator(int(i)).name.startswith('g1_b/wuji_right/r_') and
                         any(token in m.actuator(int(i)).name for token in ('THJ','FFJ'))]
        original_q=d.qpos.copy()
        def surface_gap():
            mujoco.mj_forward(m,d)
            return min(float(mujoco.mj_geomDistance(m,d,a,b,.1,np.zeros(6)))
                       for a in thumb_geoms for b in index_geoms)
        gap0=surface_gap()
        gradient=[]
        for i in pinch_actuators:
            adr=wbc.act_qadr[i]
            d.qpos[adr]+=1e-4
            gradient.append((surface_gap()-gap0)/1e-4)
            d.qpos[adr]=original_q[adr]
        gradient=np.asarray(gradient)
        delta=-args.pinch_closure_mm/1000*gradient/max(float(gradient@gradient),1e-12)
        for i,move in zip(pinch_actuators,delta):
            low,high=m.actuator_ctrlrange[i]
            wbc.hand_commands[i]=np.clip(wbc.hand_commands[i]+np.clip(move,-.15,.15),low,high)
        d.qpos[:]=original_q;mujoco.mj_forward(m,d)
if args.wrist_weight is not None:
    wbc.weights[wrist_vadr]=args.wrist_weight
interval = round(.02/m.opt.timestep)
fixture_vertices=tuple(range(m.nflexvert-2,m.nflexvert)) if args.extended_tab else (64,65)
fixture_ids=([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,f'bandage_tab_fixture_{i}') for i in fixture_vertices]
             if args.releasable_fixture else [])
duration = args.duration_s
initial_edge = d.flexvert_xpos[list(fixture_vertices)].mean(axis=0).copy()
initial_row28 = d.flexvert_xpos[56:58].mean(axis=0).copy()
screen = json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
tip_target = np.array(screen['results'][0]['target_m'])
if args.track_edge:
    wbc.tip_position_weight = 1000.
material_targets={}
initial_material_offsets={}
if args.material_tip_follow and not args.start_at_goal:
    parser.error('Material-point fingertip following currently requires --start-at-goal')
if args.material_tip_follow:
    wbc.tip_position_weight=args.tip_weight
if args.pinchtab and args.start_at_goal:
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
        initial_material_offsets[tip_name]=offset.copy()
        target_offset=offset*max(0.,1.-args.tip_closure_mm/1000/max(np.linalg.norm(offset),1e-9))
        material_targets[tip_name]=(verts,weights,target_offset)
max_pen = 0.0
max_tip_error = 0.0
feet_count = 0
flex_contact_count = 0
opposed_contact_count = 0
tip_normal_impulse_ns={'thumb':0.,'index':0.}
tip_peak_single_contact_normal_force_n={'thumb':0.,'index':0.}
tip_shear_impulse_ns={'thumb':0.,'index':0.}
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
        if not args.extended_tab:
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
        if args.pinch_bias_preset and (args.pinch_bias_until_s is None or d.time<args.pinch_bias_until_s):
            bias={'balanced_a':{'THJ1':.1,'FFJ1':.1,'FFJ3':.1},
                  'balanced_b':{'THJ1':.05,'THJ0':.05,'THJ3':-.05,'FFJ3':.05}}[args.pinch_bias_preset]
            for token,amount in bias.items():
                actuator=next(i for i in pinch_actuators if m.actuator(i).name.endswith(token))
                d.ctrl[actuator]=np.clip(d.ctrl[actuator]+amount,*m.actuator_ctrlrange[actuator])
        mujoco.mj_forward(m,d)
        local_tip_force_correction_norm=0.
        if args.measured_tip_force_n is not None:
            def measured_tip_forces():
                forces=np.zeros(2)
                for ci,c in enumerate(d.contact):
                    if not any(f>=0 for f in c.flex):continue
                    bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
                    for digit,body in enumerate(tip_names):
                        if body in bodies:
                            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                            forces[digit]+=max(0.,float(wrench[0]))
                return forces
            base=measured_tip_forces()
            columns=[]
            for actuator in pinch_actuators:
                old=float(d.ctrl[actuator])
                d.ctrl[actuator]=np.clip(old+.01,*m.actuator_ctrlrange[actuator])
                mujoco.mj_forward(m,d)
                columns.append((measured_tip_forces()-base)/max(float(d.ctrl[actuator]-old),1e-9))
                d.ctrl[actuator]=old
            mujoco.mj_forward(m,d)
            jac=np.asarray(columns).T
            if np.all(np.isfinite(jac)):
                delta=jac.T@np.linalg.solve(jac@jac.T+.01*np.eye(2),
                    np.maximum(0.,args.measured_tip_force_n-base))
                delta=np.clip(delta,-.05,.05)
                for actuator,adjustment in zip(pinch_actuators,delta):
                    d.ctrl[actuator]=np.clip(d.ctrl[actuator]+adjustment,*m.actuator_ctrlrange[actuator])
                local_tip_force_correction_norm=float(np.linalg.norm(delta))
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
            if args.fixture_release_s is not None and d.time>=args.fixture_release_s:
                d.eq_active[fixture_ids]=0
            mujoco.mj_step(m, d)
            if substep==0:
                first_step_actual_wrist_acceleration=float(d.qacc[wrist_vadr])
            if not np.all(np.isfinite(d.qpos)):
                raise RuntimeError('Nonfinite state')
            feet = {'a': set(), 'b': set()}
            has_forearm_flex = False
            tip_normals={'thumb':[],'index':[]}
            for ci,c in enumerate(d.contact):
                geoms = [int(g) for g in c.geom if g >= 0]
                names = [m.geom(g).name for g in geoms]
                bodies = [m.body(m.geom_bodyid[g]).name for g in geoms]
                flex = any(f >= 0 for f in c.flex)
                if flex and 'training_forearm_dynamic' in bodies:
                    has_forearm_flex = True
                if flex and c.dist<.003:
                    for digit,body_name in (('thumb',tip_names[0]),('index',tip_names[1])):
                        if body_name in bodies:
                            tip_normals[digit].append(c.frame[:3].copy())
                            wrench=np.zeros(6)
                            mujoco.mj_contactForce(m,d,ci,wrench)
                            normal=max(0.,float(wrench[0]))
                            tip_normal_impulse_ns[digit]+=normal*m.opt.timestep
                            tip_shear_impulse_ns[digit]+=float(np.linalg.norm(wrench[1:3]))*m.opt.timestep
                            tip_peak_single_contact_normal_force_n[digit]=max(tip_peak_single_contact_normal_force_n[digit],normal)
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
            edge_drift = max(edge_drift, float(np.linalg.norm(d.flexvert_xpos[list(fixture_vertices)].mean(axis=0)-initial_edge)))
        tip_error = float(np.max(np.abs(d.qpos[qa]-wbc.qref[qa])))
        thumb_index_surface_gap=min(float(mujoco.mj_geomDistance(m,d,a,b,.2,np.zeros(6)))
                                    for a in thumb_tip_geoms for b in index_tip_geoms)
        tip_displacements=[float(np.linalg.norm(d.xpos[bid]-initial_tip_positions[i]))
                           for i,bid in enumerate(tip_ids)]
        material_relative_shifts={name:float(np.linalg.norm(
            d.xpos[m.body(name).id]-(weights@d.flexvert_xpos[verts])-initial_material_offsets[name]))
            for name,(verts,weights,_) in material_targets.items()}
        material_relative_shift_vectors={name:(d.xpos[m.body(name).id]-
            (weights@d.flexvert_xpos[verts])-initial_material_offsets[name]).tolist()
            for name,(verts,weights,_) in material_targets.items()}
        max_tip_error = max(max_tip_error, tip_error)
        trace.append({'time_s': float(d.time), 'path_progress': float(smooth),
                      'predicted_tip_contact_force_n':{name:float(wbc.last_flex_tip_predicted_force_n.get(tip_names[i],0.)) for i,name in enumerate(('thumb','index'))},
                      'local_tip_force_correction_norm':local_tip_force_correction_norm,
                      'max_joint_tracking_error_rad': tip_error,
                      'thumb_index_surface_gap_m':thumb_index_surface_gap,
                      'tip_displacement_m':tip_displacements,
                      'tip_to_material_relative_shift_m':material_relative_shifts,
                      'tip_to_material_relative_shift_vector_m':material_relative_shift_vectors,
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
                      'free_edge_drift_m': float(np.linalg.norm(d.flexvert_xpos[list(fixture_vertices)].mean(axis=0)-initial_edge)),
                      'contact_task_residual_m_s2': meta['contact_acceleration_task_residual_m_s2']})
except Exception as exc:
    error = f'{type(exc).__name__}: {exc}'

steps = max(1, solves*interval)
report = {'scope': 'Dynamic WBC tracking of a static path toward a flexible free tab; no force-closed grasp or completed wrap',
          'track_edge_tip_targets': args.track_edge,
          'pinchtab':args.pinchtab,
          'extended_tab':args.extended_tab,
          'start_at_goal':args.start_at_goal,
          'edge_damping':args.edge_damping,
          'flex_tip_force_goal_n':args.flex_tip_force_n,
          'measured_tip_force_goal_n':args.measured_tip_force_n,
          'pinch_bias_preset':args.pinch_bias_preset,
          'pinch_bias_until_s':args.pinch_bias_until_s,
          'flex_sliding_friction':float(m.flex_friction[0,0]),
          'b_right_wrist_roll_acceleration_weight':args.wrist_weight,
          'local_physics_correction':args.local_physics_correction,
          'stabilize_other_digits':args.stabilize_other_digits,
          'stabilize_all_digits':args.stabilize_all_digits,
          'material_tip_follow':args.material_tip_follow,
          'tip_closure_mm':args.tip_closure_mm,
          'tip_position_weight':args.tip_weight if args.material_tip_follow else None,
          'flex_measured_force_prior_weight':args.flex_measured_force_prior_weight,
          'releasable_fixture':args.releasable_fixture,
          'fixture_release_s':args.fixture_release_s,
          'fixture_active_at_end':bool(all(d.eq_active[i] for i in fixture_ids)) if fixture_ids else None,
          'pinch_closure_command_mm':args.pinch_closure_mm,
          'b_right_palm_position_weight':args.b_right_palm_weight,
          'duration_s': float(d.time), 'planned_duration_s': duration, 'control_solves': solves,
          'error': error, 'two_robot_two_foot_fraction': feet_count/steps,
          'flex_forearm_contact_fraction': flex_contact_count/steps,
          'opposed_b_tip_flex_contact_fraction':opposed_contact_count/steps,
          'max_b_nonfloor_penetration_m': max_pen,
          'max_free_edge_drift_m': edge_drift,
          'mean_tip_normal_force_n':{name:impulse/max(float(d.time),1e-9) for name,impulse in tip_normal_impulse_ns.items()},
          'peak_single_contact_normal_force_n':tip_peak_single_contact_normal_force_n,
          'mean_tip_shear_force_n':{name:impulse/max(float(d.time),1e-9) for name,impulse in tip_shear_impulse_ns.items()},
          'max_tip_to_material_slip_m': max((max(item.get('tip_to_material_relative_shift_m',{}).values(),default=0.) for item in trace),default=0.),
          'max_joint_tracking_error_rad': max_tip_error,
          'final_joint_target_error_rad': float(np.max(np.abs(d.qpos[qa]-waypoints[-1, qa]))),
          'trace': trace}
report['external_end_fixture_removed_for_test'] = bool(not fixture_ids or
    (args.fixture_release_s is not None and args.fixture_release_s <= duration-.2+1e-9 and
     not any(d.eq_active[i] for i in fixture_ids)))
report['material_grasp_gate_pass'] = bool(report['external_end_fixture_removed_for_test'] and
    error is None and d.time >= duration-1e-6 and
    opposed_contact_count/steps >= .95 and edge_drift <= .003 and
    report['max_tip_to_material_slip_m'] <= .003 and max_pen <= .003)
report_name = (('joint_b_pinchtab_dynamic_path_rollout' if args.pinchtab else
                'joint_b_flex_free_edge_dynamic_path_rollout')+
               ('_goal_hold' if args.start_at_goal else '')+
               ('_extended20mm' if args.extended_tab else '')+
               ('_track_edge' if args.track_edge else '')+
               (f'_flexforce{args.flex_tip_force_n:g}' if args.flex_tip_force_n is not None else '')+
               (f'_measuredforce{args.measured_tip_force_n:g}' if args.measured_tip_force_n is not None else '')+
               (f'_pinchbias{args.pinch_bias_preset}' if args.pinch_bias_preset else '')+
               (f'_biasuntil{args.pinch_bias_until_s:g}s' if args.pinch_bias_until_s is not None else '')+
               (f'_friction{args.flex_friction:g}' if args.flex_friction is not None else '')+
               (f'_wristweight{args.wrist_weight:g}' if args.wrist_weight is not None else '')+
               ('_localphysics' if args.local_physics_correction else '')+
               ('_stabilizedigits' if args.stabilize_other_digits else '')+
               ('_stabilizealldigits' if args.stabilize_all_digits else '')+
               ('_materialfollow' if args.material_tip_follow else '')+
               (f'_closure{args.tip_closure_mm:g}mm' if args.tip_closure_mm else '')+
               (f'_tipweight{args.tip_weight:g}' if args.material_tip_follow and args.tip_weight!=1000. else '')+
               (f'_forceprior{args.flex_measured_force_prior_weight:g}' if args.flex_measured_force_prior_weight else '')+
               ('_fixture' if args.releasable_fixture else '')+
               (f'_release{args.fixture_release_s:g}s' if args.fixture_release_s is not None else '')+
               (f'_pinchclosure{args.pinch_closure_mm:g}mm' if args.pinch_closure_mm else '')+
               (f'_palmweight{args.b_right_palm_weight:g}' if args.b_right_palm_weight!=1000. else '')+
               suffix+'.json')
(OUT/report_name).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='trace'}))

