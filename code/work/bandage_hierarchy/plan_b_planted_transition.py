"""Screen a continuous planted-foot B posture change; no dynamics claim."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser = argparse.ArgumentParser()
parser.add_argument('--schedule', choices=('synchronous', 'arms_first', 'base_first', 'open_then_close'),
                    default='synchronous')
parser.add_argument('--collision-refine', action='store_true')
parser.add_argument('--stance-file',default='joint_b_planted_stance_preshape_lowered_audit.json')
args = parser.parse_args()
stance = json.loads((OUT / args.stance_file).read_text())
screen = json.loads((OUT / 'joint_b_reposition_static_screen.json').read_text())
model = mujoco.MjModel.from_xml_path(str(OUT / screen['scene']))
state = np.load(OUT / 'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz')
q0 = state['qpos'].copy()
data = mujoco.MjData(model)
data.qpos[:] = q0
mujoco.mj_forward(model, data)
feet = [model.body(f'g1_b/{s}_ankle_roll_link').id for s in ('left', 'right')]
foot_pos = [data.xpos[i].copy() for i in feet]
foot_rot = [data.xmat[i].reshape(3, 3).copy() for i in feet]
base_adr = model.joint('g1_b/floating_base').qposadr[0]
leg_names = stance['leg_joint_names']
leg_ids = [model.joint(n).id for n in leg_names]
leg_qa = np.array([model.jnt_qposadr[j] for j in leg_ids])
leg_lo = model.jnt_range[leg_ids, 0] + 1e-5
leg_hi = model.jnt_range[leg_ids, 1] - 1e-5
leg_goal = np.asarray(stance['leg_joint_targets_rad'])
arm_qa = np.array([model.joint(n).qposadr[0] for n in screen['joint_names']])
arm_goal = np.asarray(stance['source_candidate']['joint_targets_rad'])
hand = [(model.joint(n).qposadr[0], v) for n, v in stance['right_hand_precontact_targets_rad'].items()]
foot_geom = [i for i in range(model.ngeom) if model.geom(i).name.startswith('g1_b/')
             and 'official_foot_contact_' in model.geom(i).name]
b_bodies = [i for i in range(model.nbody) if model.body(i).name.startswith('g1_b/')]
b_mass = model.body_mass[b_bodies]
left_wrist = model.body('g1_b/wuji_left/l_wrist').id
right_wrist = model.body('g1_b/wuji_right/r_wrist').id
left_start = data.xpos[left_wrist].copy()
right_goal = np.asarray(stance['source_candidate']['right_goal_world_m'])
right_start = data.xpos[right_wrist].copy()
right_arm_names = [n for n in screen['joint_names'] if n.startswith('g1_b/right_')]
right_arm_ids = [model.joint(n).id for n in right_arm_names]
right_arm_qa = np.array([model.jnt_qposadr[j] for j in right_arm_ids])
right_arm_lo = model.jnt_range[right_arm_ids,0]+1e-5
right_arm_hi = model.jnt_range[right_arm_ids,1]-1e-5

def evaluate(alpha, leg_seed):
    if args.schedule == 'arms_first':
        base_phase, arm_phase, hand_phase = max(0., 2*alpha-1), min(1., 2*alpha), alpha
    elif args.schedule == 'base_first':
        base_phase, arm_phase, hand_phase = min(1., 2*alpha), max(0., 2*alpha-1), alpha
    elif args.schedule == 'open_then_close':
        base_phase, arm_phase, hand_phase = min(1., 1.5*alpha), min(1., 1.5*alpha), max(0., 3*alpha-2)
    else:
        base_phase = arm_phase = hand_phase = alpha
    q = q0.copy()
    q[base_adr] += base_phase * stance['base_x_shift_m']
    q[base_adr + 2] += base_phase * stance['base_z_shift_m']
    q[arm_qa] = q0[arm_qa] + arm_phase * (arm_goal - q0[arm_qa])
    for adr, target in hand:
        q[adr] = q0[adr] + hand_phase * (target - q0[adr])
    def residual(x):
        data.qpos[:] = q
        data.qpos[leg_qa] = x
        mujoco.mj_forward(model, data)
        pieces = []
        for bid, pos, rot in zip(feet, foot_pos, foot_rot):
            pieces.extend((data.xpos[bid] - pos, .06 * (data.xmat[bid].reshape(3, 3) - rot).ravel()))
        pieces.append(.0001 * (x - leg_seed))
        return np.concatenate(pieces)
    fit = least_squares(residual, np.clip(leg_seed, leg_lo, leg_hi),
                        bounds=(leg_lo, leg_hi), max_nfev=250)
    residual(fit.x)
    if args.collision_refine and alpha not in (0., 1.):
        arm_seed = data.qpos[right_arm_qa].copy()
        wrist_target = right_start + alpha * (right_goal-right_start)
        def arm_residual(x):
            data.qpos[right_arm_qa] = x
            mujoco.mj_forward(model, data)
            penetration = 0.
            for contact in data.contact:
                bodies = [model.body(model.geom_bodyid[g]).name for g in contact.geom]
                if any(n.startswith('g1_b/wuji_right/') for n in bodies) and \
                   'training_forearm_dynamic' in bodies:
                    penetration = max(penetration, -float(contact.dist))
            return np.r_[data.xpos[right_wrist]-wrist_target,
                         15.*max(0., penetration-.001), .004*(x-arm_seed)]
        arm_fit=least_squares(arm_residual,np.clip(arm_seed,right_arm_lo,right_arm_hi),
                              bounds=(right_arm_lo,right_arm_hi),max_nfev=100)
        arm_residual(arm_fit.x)
    foot_err = [float(np.linalg.norm(data.xpos[bid] - pos)) for bid, pos in zip(feet, foot_pos)]
    foot_rot_err = [float(np.arccos(np.clip((np.trace(data.xmat[bid].reshape(3,3).T @ rot)-1)/2,-1,1)))
                    for bid, rot in zip(feet, foot_rot)]
    com = (b_mass[:, None] * data.xipos[b_bodies]).sum(axis=0) / b_mass.sum()
    hull = ConvexHull(data.geom_xpos[foot_geom, :2])
    margin = float(min(-(e[0]*com[0]+e[1]*com[1]+e[2])/np.linalg.norm(e[:2]) for e in hull.equations))
    deep = []
    floor_sides = set()
    for c in data.contact:
        geoms = [model.geom(g).name for g in c.geom]
        bodies = [model.body(model.geom_bodyid[g]).name for g in c.geom]
        if 'floor' in geoms:
            for side in ('left', 'right'):
                if any(n.startswith(f'g1_b/{side}_official_foot_contact_') for n in geoms):
                    floor_sides.add(side)
        elif c.dist < -.003 and any(n.startswith('g1_b/') for n in bodies):
            deep.append({'geoms': geoms, 'bodies': bodies, 'penetration_m': -float(c.dist)})
    return fit.x, {'phase': float(alpha), 'base_phase':float(base_phase),
                   'arm_phase':float(arm_phase), 'hand_phase':float(hand_phase),
                   'foot_position_errors_m': foot_err,
                   'foot_rotation_errors_rad': foot_rot_err, 'floor_contact_sides': sorted(floor_sides),
                   'com_support_margin_m': margin,
                   'left_wrist_drift_m': float(np.linalg.norm(data.xpos[left_wrist]-left_start)),
                   'right_wrist_goal_error_m': float(np.linalg.norm(data.xpos[right_wrist]-right_goal)),
                   'right_wrist_path_error_m': float(np.linalg.norm(
                       data.xpos[right_wrist]-(right_start+alpha*(right_goal-right_start)))),
                   'deep_nonfloor_contacts': deep,
                   'leg_limit_margin_min_rad': float(np.min(np.minimum(fit.x-leg_lo,leg_hi-fit.x))),
                   'qpos': data.qpos.tolist()}

seed = q0[leg_qa].copy()
rows = []
for alpha in np.linspace(0, 1, 21):
    seed, row = evaluate(alpha, seed)
    rows.append(row)
joint_steps = [float(np.max(np.abs(np.asarray(b['qpos'])-np.asarray(a['qpos']))))
               for a,b in zip(rows,rows[1:])]
dense_max_penetration = 0.
dense_deep_samples = 0
dense_max_foot_error = 0.
for a,b in zip(rows,rows[1:]):
    q_a=np.asarray(a['qpos']);q_b=np.asarray(b['qpos'])
    for t in np.linspace(0,1,11)[1:-1]:
        data.qpos[:] = (1-t)*q_a+t*q_b
        mujoco.mj_forward(model,data)
        dense_max_foot_error=max(dense_max_foot_error,max(float(np.linalg.norm(data.xpos[bid]-pos))
            for bid,pos in zip(feet,foot_pos)))
        current_pen=0.
        for c in data.contact:
            names=[model.geom(g).name for g in c.geom]
            bodies=[model.body(model.geom_bodyid[g]).name for g in c.geom]
            if 'floor' not in names and any(n.startswith('g1_b/') for n in bodies):
                current_pen=max(current_pen,-float(c.dist))
        dense_max_penetration=max(dense_max_penetration,current_pen)
        dense_deep_samples+=current_pen>.003
report = {'scope': 'Kinematic planted-foot interpolation only; no time parameterization or WBC integration',
          'schedule': args.schedule, 'collision_refine': args.collision_refine,
          'source_stance':args.stance_file,
          'samples': rows,
          'all_foot_pose_below_2mm': all(max(r['foot_position_errors_m']) < .002 for r in rows),
          'all_two_feet_contact': all(r['floor_contact_sides'] == ['left','right'] for r in rows),
          'minimum_com_support_margin_m': min(r['com_support_margin_m'] for r in rows),
          'maximum_nonfloor_penetration_m': max((c['penetration_m'] for r in rows for c in r['deep_nonfloor_contacts']), default=0.),
          'maximum_joint_step_rad_or_m':max(joint_steps),
          'maximum_right_wrist_path_error_m':max(r['right_wrist_path_error_m'] for r in rows),
          'dense_interpolated_max_nonfloor_penetration_m':dense_max_penetration,
          'dense_interpolated_deep_sample_count':dense_deep_samples,
          'dense_interpolated_max_foot_position_error_m':dense_max_foot_error}
suffix='_lowered10mm' if 'lowered10mm' in args.stance_file else ''
(OUT / f'joint_b_planted_transition_{args.schedule}{"_refined" if args.collision_refine else ""}{suffix}_kinematic_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k!='samples'} | {
    'max_foot_error_m': max(max(r['foot_position_errors_m']) for r in rows),
    'max_left_wrist_drift_m': max(r['left_wrist_drift_m'] for r in rows),
    'samples_with_deep_contact': sum(bool(r['deep_nonfloor_contacts']) for r in rows)}))
