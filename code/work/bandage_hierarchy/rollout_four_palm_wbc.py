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
for _ in range(round(.1/model.opt.timestep)):
    mujoco.mj_step(model, data)
palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
def robot_contact_only(model, contact):
    return any(model.body(model.geom_bodyid[int(g)]).name.startswith(('g1_a/', 'g1_b/'))
               for g in contact.geom if g >= 0)


wbc = JointWBC(model, data.qpos.copy(), palm_names=palms,
               contact_filter=robot_contact_only)
root_ids = [model.body(f'{robot}/pelvis').id for robot in ('g1_a', 'g1_b')]
initial_roots = data.xpos[root_ids].copy()
control_interval = round((1/args.hz)/model.opt.timestep)
expected_solves = round(.3*args.hz)
support_frames = 0
max_root_motion = 0.
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
        break
report = {'scope': '0.3 s closed-loop stance after 0.1 s uncontrolled settle; no cloth or bandaging motion',
          'wbc_hz': args.hz, 'solves_completed': solves, 'expected_solves': expected_solves,
          'two_robot_foot_support_fraction': support_frames/(expected_solves*control_interval),
          'max_pelvis_displacement_m': max_root_motion,
          'finite_state': bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
          'error': error, 'failure_snapshot': failure_snapshot}
(OUT / f'four_palm_wbc_stance_{args.hz}hz.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
if error or solves != expected_solves:
    raise RuntimeError(error or 'Incomplete stance rollout')
