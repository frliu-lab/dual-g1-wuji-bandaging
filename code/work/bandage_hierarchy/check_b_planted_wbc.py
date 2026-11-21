"""One-frame inverse-dynamics feasibility probe; not a rollout."""
import json
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser = argparse.ArgumentParser()
parser.add_argument('--rollout-s', type=float, default=0.)
args = parser.parse_args()
sys.path.insert(0, str(ROOT / 'work/wuji_multiagent'))
from joint_wbc import JointWBC

stance = json.loads((OUT / 'joint_b_planted_stance_preshape_lowered_audit.json').read_text())
screen = json.loads((OUT / 'joint_b_reposition_static_screen.json').read_text())
model = mujoco.MjModel.from_xml_path(str(OUT / screen['scene']))
state = np.load(OUT / 'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz')
data = mujoco.MjData(model)
data.qpos[:] = state['qpos']
base_adr = model.joint('g1_b/floating_base').qposadr[0]
data.qpos[base_adr] += stance['base_x_shift_m']
data.qpos[base_adr + 2] += stance['base_z_shift_m']
for name, val in zip(screen['joint_names'], stance['source_candidate']['joint_targets_rad']):
    data.qpos[model.joint(name).qposadr[0]] = val
for name, val in zip(stance['leg_joint_names'], stance['leg_joint_targets_rad']):
    data.qpos[model.joint(name).qposadr[0]] = val
for name, val in stance['right_hand_precontact_targets_rad'].items():
    data.qpos[model.joint(name).qposadr[0]] = val
data.qvel[:] = 0
mujoco.mj_forward(model, data)

def robot_contact_only(m, contact):
    return any(m.body(m.geom_bodyid[int(g)]).name.startswith(('g1_a/', 'g1_b/'))
               for g in contact.geom if g >= 0)

palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
wbc = JointWBC(model, data.qpos.copy(), palm_names=palms,
               contact_filter=robot_contact_only)
report = {'scope': ('Directly initialized planted stance rollout; no physical transition into stance or bandaging'
                    if args.rollout_s else 'Single-state inverse-dynamics QP only; no integration, transition, or contact-force validation'),
          'source_stance': 'joint_b_planted_stance_preshape_lowered_audit.json',
          'contact_count': int(data.ncon), 'solve_success': False}
try:
    control, meta, _ = wbc.solve(data)
    report['solve_success'] = True
    report['finite_control'] = bool(np.all(np.isfinite(control)))
    report['control_range_margin_min'] = float(min(
        min(control[i] - model.actuator_ctrlrange[i, 0],
            model.actuator_ctrlrange[i, 1] - control[i])
        for i in range(model.nu) if model.actuator_ctrllimited[i]))
    report['max_abs_control'] = float(np.max(np.abs(control)))
    report['meta'] = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
    if args.rollout_s > 0:
        step_interval = round(.02 / model.opt.timestep)
        steps = round(args.rollout_s / .02)
        root_id = model.body('g1_b/pelvis').id
        root_start = data.xpos[root_id].copy()
        foot_ids = [i for i in range(model.ngeom) if model.geom(i).name.startswith('g1_b/')
                    and 'official_foot_contact_' in model.geom(i).name]
        foot_start = data.geom_xpos[foot_ids].copy()
        root_drift = foot_drift = 0.
        max_deep_nonfloor = 0.
        right_wrist_id = model.body('g1_b/wuji_right/r_wrist').id
        left_wrist_id = model.body('g1_b/wuji_left/l_wrist').id
        left_start = data.xpos[left_wrist_id].copy()
        right_goal = np.asarray(stance['source_candidate']['right_goal_world_m'])
        right_error = 0.
        left_drift = 0.
        support_frames = 0
        for _ in range(steps):
            control, meta, _ = wbc.solve(data)
            data.ctrl[:] = control
            for _ in range(step_interval):
                mujoco.mj_step(model, data)
                root_drift = max(root_drift, float(np.linalg.norm(data.xpos[root_id]-root_start)))
                foot_drift = max(foot_drift, float(np.max(np.linalg.norm(
                    data.geom_xpos[foot_ids, :2]-foot_start[:, :2], axis=1))))
                right_error = max(right_error, float(np.linalg.norm(data.xpos[right_wrist_id]-right_goal)))
                left_drift = max(left_drift, float(np.linalg.norm(data.xpos[left_wrist_id]-left_start)))
                for c in data.contact:
                    names = [model.geom(g).name for g in c.geom]
                    if 'floor' not in names and any(model.body(model.geom_bodyid[g]).name.startswith('g1_b/') for g in c.geom):
                        max_deep_nonfloor = max(max_deep_nonfloor, -float(c.dist))
                floor_sides = set()
                for c in data.contact:
                    names = [model.geom(g).name for g in c.geom]
                    if 'floor' in names:
                        for side in ('left', 'right'):
                            if any(n.startswith(f'g1_b/{side}_official_foot_contact_') for n in names):
                                floor_sides.add(side)
                support_frames += floor_sides == {'left', 'right'}
        report['rollout'] = {'duration_s': float(data.time), 'control_solves': steps,
                             'two_foot_support_fraction': support_frames/(steps*step_interval),
                             'max_b_pelvis_drift_m': root_drift,
                             'max_b_foot_marker_xy_drift_m': foot_drift,
                             'max_b_right_wrist_goal_error_m': right_error,
                             'max_b_left_wrist_drift_m': left_drift,
                             'max_b_nonfloor_penetration_m': max_deep_nonfloor,
                             'finite_state': bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))}
except Exception as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
(OUT / ('joint_b_planted_stance_wbc_rollout_audit.json' if args.rollout_s else
        'joint_b_planted_stance_wbc_one_frame_audit.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
