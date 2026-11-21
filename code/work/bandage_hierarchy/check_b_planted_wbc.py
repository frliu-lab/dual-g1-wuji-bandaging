"""One-frame inverse-dynamics feasibility probe; not a rollout."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
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
report = {'scope': 'Single-state inverse-dynamics QP only; no integration, transition, or contact-force validation',
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
except Exception as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
(OUT / 'joint_b_planted_stance_wbc_one_frame_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
