"""Static compatibility screen for A-left palm support and A-right pad pinch.

This combines two previously isolated variants without asserting a path or grasp.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
tree = ET.parse(OUT / 'dynamic_forearm_variant.xml')
root = tree.getroot()
root.find(".//body[@name='g1_a/pelvis']").set('pos', '-.3 0 .8')
for name, size in (('dressing_pad_contact', '.045 .03 .008'),
                   ('dressing_stage', '.04 .015 .006')):
    next(g for g in root.iter('geom') if g.get('name') == name).set('size', size)
variant = OUT / 'joint_support_pinch_variant.xml'
tree.write(variant, encoding='utf8', xml_declaration=True)
model = mujoco.MjModel.from_xml_path(str(variant))
data = mujoco.MjData(model)
base = np.load(OUT / 'dynamic_palm_support_rrt.npz')['qpos'][-1].copy()
old = mujoco.MjModel.from_xml_path(str(OUT / 'pad_side_pinch_variant.xml'))
old_q = np.load(OUT / 'pad_side_pinch_candidates.npz')['qpos'][1]
transfer = []
for j in range(old.njnt):
    name = old.joint(j).name
    if not (name.startswith('g1_a/right_') and any(x in name for x in ('shoulder', 'elbow', 'wrist'))
            or name.startswith(('g1_a/wuji_right/r_thumb_', 'g1_a/wuji_right/r_index_finger_'))):
        continue
    k = model.joint(name).id
    base[model.jnt_qposadr[k]] = old_q[old.jnt_qposadr[j]]
    transfer.append(name)
data.qpos[:] = base
mujoco.mj_forward(model, data)
deep = []
touches = []
for c in data.contact:
    if c.geom1 < 0 or c.geom2 < 0:
        continue
    geoms = (model.geom(c.geom1).name, model.geom(c.geom2).name)
    bodies = (model.body(model.geom_bodyid[c.geom1]).name,
              model.body(model.geom_bodyid[c.geom2]).name)
    if 'floor' not in geoms and any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies):
        if c.dist < -.003:
            deep.append({'geoms': geoms, 'bodies': bodies, 'distance_m': float(c.dist)})
        if 'manikin_forearm' in geoms or 'dressing_pad_contact' in geoms:
            touches.append({'geoms': geoms, 'bodies': bodies, 'distance_m': float(c.dist)})
report = {
    'scope': 'Static composite screen only: no approach path, dynamic grasp, four-hand cooperation, or safety claim',
    'variant': variant.name,
    'a_root_x_m': -.3,
    'forearm_mass_kg': .8,
    'transferred_right_joints': transfer,
    'deep_nonfloor_robot_contacts': deep,
    'task_contacts': touches,
    'a_left_wrist_m': data.xpos[model.body('g1_a/wuji_left/l_wrist').id].tolist(),
    'a_right_thumb_tip_m': data.xpos[model.body('g1_a/wuji_right/r_thumb_tip_sensor_frame').id].tolist(),
    'a_right_index_tip_m': data.xpos[model.body('g1_a/wuji_right/r_index_finger_tip_sensor_frame').id].tolist(),
    'pad_center_m': data.xpos[model.body('dressing_pad').id].tolist(),
    'finite_state': bool(np.all(np.isfinite(data.qpos))),
}
joint_ids = [model.joint(name).id for name in transfer]
qadr = np.array([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
vadr = np.array([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
lo = np.array([model.jnt_range[j, 0] + .03 for j in joint_ids])
hi = np.array([model.jnt_range[j, 1] - .03 for j in joint_ids])
start = np.clip(base[qadr], lo + 1e-5, hi - 1e-5)
tip_ids = [model.body('g1_a/wuji_right/r_index_finger_tip_sensor_frame').id,
           model.body('g1_a/wuji_right/r_thumb_tip_sensor_frame').id]
goals = np.array([[-.38, -.48, .698], [-.41, -.405, .698]])

def forward(x):
    data.qpos[:] = base
    data.qpos[qadr] = x
    mujoco.mj_forward(model, data)

def residual(x):
    forward(x)
    return np.r_[30 * (data.xpos[tip_ids] - goals).ravel(), .05 * (x - start)]

def jacobian(x):
    forward(x)
    J = np.zeros((6 + len(x), len(x)))
    for i, bid in enumerate(tip_ids):
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, data.xpos[bid], bid)
        J[3*i:3*i+3] = 30 * jac[:, vadr]
    J[6:] = .05 * np.eye(len(x))
    return J

opt = least_squares(residual, start, jac=jacobian, bounds=(lo, hi), max_nfev=150)
forward(opt.x)
refined_deep = []
digit_contacts = {'thumb': [], 'index': []}
digit_normals = {'thumb': np.zeros(3), 'index': np.zeros(3)}
for ci, c in enumerate(data.contact):
    if c.geom1 < 0 or c.geom2 < 0:
        continue
    geoms = (model.geom(c.geom1).name, model.geom(c.geom2).name)
    bodies = (model.body(model.geom_bodyid[c.geom1]).name,
              model.body(model.geom_bodyid[c.geom2]).name)
    if 'floor' not in geoms and any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies) and c.dist < -.003:
        refined_deep.append({'geoms': geoms, 'bodies': bodies, 'distance_m': float(c.dist)})
    if 'dressing_pad_contact' in geoms:
        for digit in digit_contacts:
            if any(b.startswith(f'g1_a/wuji_right/r_{digit}_') for b in bodies):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, wrench)
                force = max(0., float(wrench[0]))
                digit_contacts[digit].append({'bodies': bodies, 'distance_m': float(c.dist), 'force_n': force})
                sign = 1. if geoms[0] == 'dressing_pad_contact' else -1.
                digit_normals[digit] += sign * force * np.asarray(c.frame[:3])
a, b = digit_normals.values()
normal_cosine = (float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                 if np.linalg.norm(a) * np.linalg.norm(b) > 1e-8 else None)
report['joint_ik'] = {
    'solver_success': bool(opt.success), 'solver_nfev': int(opt.nfev),
    'goals_m': goals.tolist(),
    'tip_errors_m': np.linalg.norm(data.xpos[tip_ids] - goals, axis=1).tolist(),
    'deep_nonfloor_robot_contacts': refined_deep,
    'digit_contacts': digit_contacts,
    'normal_cosine': normal_cosine,
    'eligible_static_endpoint': bool(not refined_deep and all(digit_contacts.values())
                                     and normal_cosine is not None and normal_cosine < -.5),
}
def collision_aware_objective(x):
    forward(x)
    error = float(np.sum((data.xpos[tip_ids] - goals)**2))
    penetration = 0.
    for c in data.contact:
        if c.geom1 < 0 or c.geom2 < 0:
            continue
        geoms = (model.geom(c.geom1).name, model.geom(c.geom2).name)
        bodies = (model.body(model.geom_bodyid[c.geom1]).name,
                  model.body(model.geom_bodyid[c.geom2]).name)
        if 'floor' not in geoms and any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies):
            penetration += max(0., -float(c.dist) - .002)**2
    return 1e4 * error + 2e5 * penetration + .002 * float(np.sum((x - start)**2))

refine = minimize(collision_aware_objective, opt.x, method='L-BFGS-B',
                  bounds=list(zip(lo, hi)), options={'maxiter': 100, 'ftol': 1e-8})
forward(refine.x)
refined_deep = []
refined_normals = {'thumb': np.zeros(3), 'index': np.zeros(3)}
refined_forces = {'thumb': 0., 'index': 0.}
for ci, c in enumerate(data.contact):
    if c.geom1 < 0 or c.geom2 < 0:
        continue
    geoms = (model.geom(c.geom1).name, model.geom(c.geom2).name)
    bodies = (model.body(model.geom_bodyid[c.geom1]).name,
              model.body(model.geom_bodyid[c.geom2]).name)
    if 'floor' not in geoms and any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies) and c.dist < -.003:
        refined_deep.append({'geoms': geoms, 'bodies': bodies, 'distance_m': float(c.dist)})
    if 'dressing_pad_contact' in geoms:
        for digit in refined_normals:
            if any(b.startswith(f'g1_a/wuji_right/r_{digit}_') for b in bodies):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, wrench)
                force = max(0., float(wrench[0]))
                refined_forces[digit] += force
                sign = 1. if geoms[0] == 'dressing_pad_contact' else -1.
                refined_normals[digit] += sign * force * np.asarray(c.frame[:3])
a, b = refined_normals.values()
refined_cosine = (float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                  if np.linalg.norm(a) * np.linalg.norm(b) > 1e-8 else None)
report['collision_refined'] = {
    'solver_success': bool(refine.success), 'solver_iterations': int(refine.nit),
    'tip_errors_m': np.linalg.norm(data.xpos[tip_ids] - goals, axis=1).tolist(),
    'deep_nonfloor_robot_contacts': refined_deep,
    'digit_forces_n': refined_forces,
    'normal_cosine': refined_cosine,
    'eligible_static_endpoint': bool(not refined_deep and all(f > .05 for f in refined_forces.values())
                                     and refined_cosine is not None and refined_cosine < -.5),
}
if report['collision_refined']['eligible_static_endpoint']:
    np.savez_compressed(OUT / 'joint_support_pinch_static_candidate.npz', qpos=data.qpos.copy())
scan = []
best_eligible = None
for index_y, thumb_y, z in ((-.475, -.405, .698), (-.485, -.405, .698),
                            (-.48, -.41, .698), (-.48, -.415, .698),
                            (-.48, -.405, .702), (-.48, -.405, .694)):
    goals = np.array([[-.38, index_y, z], [-.41, thumb_y, z]])
    trial = minimize(collision_aware_objective, refine.x, method='L-BFGS-B',
                     bounds=list(zip(lo, hi)), options={'maxiter': 75, 'ftol': 1e-8})
    forward(trial.x)
    trial_deep = 0
    trial_normals = {'thumb': np.zeros(3), 'index': np.zeros(3)}
    trial_forces = {'thumb': 0., 'index': 0.}
    for ci, c in enumerate(data.contact):
        if c.geom1 < 0 or c.geom2 < 0:
            continue
        geoms = (model.geom(c.geom1).name, model.geom(c.geom2).name)
        bodies = (model.body(model.geom_bodyid[c.geom1]).name,
                  model.body(model.geom_bodyid[c.geom2]).name)
        if 'floor' not in geoms and any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies) and c.dist < -.003:
            trial_deep += 1
        if 'dressing_pad_contact' in geoms:
            for digit in trial_normals:
                if any(b.startswith(f'g1_a/wuji_right/r_{digit}_') for b in bodies):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, wrench)
                    force = max(0., float(wrench[0]))
                    trial_forces[digit] += force
                    sign = 1. if geoms[0] == 'dressing_pad_contact' else -1.
                    trial_normals[digit] += sign * force * np.asarray(c.frame[:3])
    a, b = trial_normals.values()
    cosine = (float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
              if np.linalg.norm(a) * np.linalg.norm(b) > 1e-8 else None)
    eligible = bool(trial_deep == 0 and all(f > .05 for f in trial_forces.values())
                    and cosine is not None and cosine < -.5)
    row = {'goals_m': goals.tolist(), 'tip_errors_m': np.linalg.norm(data.xpos[tip_ids] - goals, axis=1).tolist(),
           'deep_contact_count': trial_deep, 'normal_cosine': cosine,
           'digit_forces_n': trial_forces, 'solver_success': bool(trial.success),
           'eligible_static_endpoint': eligible}
    scan.append(row)
    if eligible and (best_eligible is None or sum(row['tip_errors_m']) < best_eligible[0]):
        best_eligible = (sum(row['tip_errors_m']), data.qpos.copy(), row)
if best_eligible is not None:
    np.savez_compressed(OUT / 'joint_support_pinch_static_candidate.npz', qpos=best_eligible[1])
report['goal_scan'] = scan
report['best_eligible_goal_scan'] = best_eligible[2] if best_eligible is not None else None
(OUT / 'joint_support_pinch_static_screen.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in report.items() if k not in ('transferred_right_joints', 'task_contacts', 'deep_nonfloor_robot_contacts')} |
                 {'deep_count': len(deep), 'joint_ik': {k: v for k, v in report['joint_ik'].items()
                                                    if k not in ('digit_contacts', 'deep_nonfloor_robot_contacts')},
                  'collision_refined': {k: v for k, v in report['collision_refined'].items()
                                        if k != 'deep_nonfloor_robot_contacts'},
                  'eligible_scan_count': sum(x['eligible_static_endpoint'] for x in scan),
                  'best_eligible_goal_scan': report['best_eligible_goal_scan']}))
