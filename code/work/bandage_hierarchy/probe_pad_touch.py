"""Search A-right fingertip pad contact geometry without claiming a grasp."""
import json
from pathlib import Path

import mujoco
import numpy as np

from scene_evidence import hand_contacts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
m = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
d = mujoco.MjData(m)
seed = np.load(OUT / 'reach_ik_seed.npz')
base = seed['qpos'].copy()
body = m.body('g1_a/wuji_right/r_index_finger_tip_sensor_frame').id
joints = [j for j in range(m.njnt) if m.joint(j).name.startswith('g1_a/right_')
          and any(t in m.joint(j).name for t in ('shoulder','elbow','wrist'))
          and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
qa = np.array([m.jnt_qposadr[j] for j in joints], dtype=int)
va = np.array([m.jnt_dofadr[j] for j in joints], dtype=int)
results = []
for z in np.arange(.719, .7149, -.0002):
    d.qpos[:] = base
    target = np.array([-.38, -.44, float(z)])
    for _ in range(160):
        mujoco.mj_forward(m, d)
        error = target-d.xpos[body]
        if np.linalg.norm(error) < .0005:
            break
        jac = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jac, None, d.xpos[body], body)
        J = jac[:, va]
        delta = J.T @ np.linalg.solve(J@J.T+.02**2*np.eye(3), error)
        d.qpos[qa] += np.clip(delta, -.02, .02)
        for j, adr in zip(joints, qa):
            if m.jnt_limited[j]:
                low, high = m.jnt_range[j]
                d.qpos[adr] = np.clip(d.qpos[adr], low+.03, high-.03)
    mujoco.mj_forward(m, d)
    deep = []
    pad_contacts = []
    for i, c in enumerate(d.contact):
        if c.geom1 < 0 or c.geom2 < 0:
            continue
        names = [m.geom(g).name for g in (c.geom1,c.geom2)]
        bodies = [m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
        if 'floor' not in names and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
            if c.dist < -.003:
                deep.append({'bodies': bodies, 'distance_m': float(c.dist)})
            if 'dressing_pad_contact' in names:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, wrench)
                pad_contacts.append({'bodies': bodies, 'distance_m': float(c.dist),
                                     'normal_force_n': float(wrench[0])})
    results.append({'target_z_m': float(z),
                    'tip_error_m': float(np.linalg.norm(target-d.xpos[body])),
                    'deep_robot_contacts': deep, 'pad_contacts': pad_contacts,
                    'four_hand_evidence': hand_contacts(m, d)})
report = {'scope': 'Static A-right fingertip vertical contact scan; no dynamic approach or grasp',
          'candidates': results}
(OUT / 'pad_touch_scan.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps([{'z': round(x['target_z_m'],3), 'pad': len(x['pad_contacts']),
                   'deep': len(x['deep_robot_contacts']), 'evidence': x['four_hand_evidence'][1]}
                  for x in results]))
