"""Short two-G1 four-palm stance rollout in the bandaging station.

No cloth, target contacts or bandaging motion are attempted here.
"""
import json
import argparse
from pathlib import Path
import sys
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser = argparse.ArgumentParser()
parser.add_argument('--hz', type=int, default=200)
parser.add_argument('--duration', type=float, default=.3)
parser.add_argument('--init-mode', choices=('contact', 'drop'), default='contact')
args = parser.parse_args()
sys.path.insert(0, str(ROOT / 'work/wuji_multiagent'))
from joint_wbc import JointWBC


def floor_support(model, data):
    seen = set()
    for c in data.contact:
        ids = (int(c.geom1), int(c.geom2))
        if min(ids) < 0:
            continue
        names = [model.geom(i).name for i in ids]
        if 'floor' not in names:
            continue
        robot_id = ids[1-names.index('floor')]
        body_name = model.body(model.geom_bodyid[robot_id]).name
        if body_name.startswith('g1_a/'):
            seen.add('g1_a')
        if body_name.startswith('g1_b/'):
            seen.add('g1_b')
    return seen


model = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
data = mujoco.MjData(model)
if args.init_mode == 'drop':
    for _ in range(round(.1/model.opt.timestep)):
        mujoco.mj_step(model, data)
else:
    mujoco.mj_forward(model, data)
    for robot in ('g1_a', 'g1_b'):
        foot_geoms = [i for i in range(model.ngeom)
                      if model.geom(i).name.startswith(robot+'/')
                      and 'official_foot_contact' in model.geom(i).name]
        clearance = min(float(data.geom_xpos[i, 2]-model.geom_size[i, 0])
                        for i in foot_geoms)
        root_joint = model.joint(f'{robot}/floating_base').id
        data.qpos[model.jnt_qposadr[root_joint]+2] -= clearance+.0002
    mujoco.mj_forward(model, data)
palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
def robot_contact_only(model, contact):
    return any(model.body(model.geom_bodyid[int(g)]).name.startswith(('g1_a/', 'g1_b/'))
               for g in contact.geom if g >= 0)


wbc = JointWBC(model, data.qpos.copy(), palm_names=palms,
               contact_filter=robot_contact_only)
root_ids = [model.body(f'{robot}/pelvis').id for robot in ('g1_a', 'g1_b')]
initial_roots = data.xpos[root_ids].copy()
foot_geom_ids = [i for i in range(model.ngeom)
                 if 'official_foot_contact' in model.geom(i).name]
initial_feet = data.geom_xpos[foot_geom_ids].copy()
limited_joints = [j for j in range(model.njnt)
                  if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[j]]
ankle_dofs = [int(model.jnt_dofadr[j]) for j in limited_joints
              if 'ankle' in model.joint(j).name]
control_interval = round((1/args.hz)/model.opt.timestep)
expected_solves = round(args.duration*args.hz)
support_frames = 0
max_root_motion = 0.
max_foot_planar_motion = 0.
max_ankle_speed = 0.
min_joint_limit_margin = float('inf')
solves = 0
error = None
failure_snapshot = None
for _ in range(expected_solves):
    try:
        ctrl, meta, _ = wbc.solve(data)
        data.ctrl[:] = ctrl
        solves += 1
        for _ in range(control_interval):
            mujoco.mj_step(model, data)
            support_frames += int(floor_support(model, data) == {'g1_a', 'g1_b'})
            max_root_motion = max(max_root_motion,
                                  float(np.max(np.linalg.norm(data.xpos[root_ids]-initial_roots, axis=1))))
            max_foot_planar_motion = max(max_foot_planar_motion,
                                         float(np.max(np.linalg.norm(
                                             data.geom_xpos[foot_geom_ids, :2]-initial_feet[:, :2], axis=1))))
            max_ankle_speed = max(max_ankle_speed, float(np.max(np.abs(data.qvel[ankle_dofs]))))
            for j in limited_joints:
                q = data.qpos[model.jnt_qposadr[j]]
                low, high = model.jnt_range[j]
                min_joint_limit_margin = min(min_joint_limit_margin, float(q-low), float(high-q))
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        contacts = []
        for c in data.contact:
            if c.geom1 >= 0 and c.geom2 >= 0:
                contacts.append((model.geom(c.geom1).name,
                                 model.geom(c.geom2).name, float(c.dist)))
        failure_snapshot = {'time_s': float(data.time), 'contact_count': data.ncon,
                            'floor_support': sorted(floor_support(model, data)),
                            'deepest_contacts': sorted(contacts, key=lambda x: x[2])[:12]}
        numeric = getattr(wbc, 'last_numeric', None)
        if numeric is not None:
            inversion = numeric['acc_lo']-numeric['acc_hi']
            bad_dofs = np.flatnonzero(inversion > 1e-9)
            failure_snapshot['max_acc_bound_inversion'] = float(max(0., inversion.max()))
            failure_snapshot['bad_acceleration_dofs'] = [
                {'joint': model.joint(model.dof_jntid[int(i)]).name,
                 'inversion_m_s2_or_rad_s2': float(inversion[i]),
                 'q_rad': float(data.qpos[model.jnt_qposadr[model.dof_jntid[int(i)]]]),
                 'v_rad_s': float(data.qvel[int(i)]),
                 'joint_range_rad': model.jnt_range[model.dof_jntid[int(i)]].tolist(),
                 'acc_lo': float(numeric['acc_lo'][i]),
                 'acc_hi': float(numeric['acc_hi'][i])}
                for i in bad_dofs[:20]]
            failure_snapshot['contact_constraints_in_qp'] = len(numeric['mu'])
        break
report = {'scope': 'Closed-loop stance only; no cloth or bandaging motion',
          'initialization': args.init_mode,
          'target_duration_s': args.duration,
          'wbc_hz': args.hz, 'solves_completed': solves, 'expected_solves': expected_solves,
          'two_robot_foot_support_fraction': support_frames/(expected_solves*control_interval),
          'max_pelvis_displacement_m': max_root_motion,
          'max_foot_contact_marker_planar_motion_m': max_foot_planar_motion,
          'max_ankle_speed_rad_s': max_ankle_speed,
          'minimum_joint_limit_margin_rad': min_joint_limit_margin,
          'finite_state': bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
          'error': error, 'failure_snapshot': failure_snapshot}
(OUT / f'four_palm_wbc_stance_{args.hz}hz_{args.init_mode}_{args.duration:g}s.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
if error or solves != expected_solves:
    raise RuntimeError(error or 'Incomplete stance rollout')
