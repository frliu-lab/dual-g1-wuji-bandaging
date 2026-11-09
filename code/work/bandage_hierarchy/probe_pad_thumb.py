"""Scan Wuji thumb positions near a dressing pad edge; static geometry only."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
m = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
d = mujoco.MjData(m)
base = np.load(OUT / 'reach_ik_seed.npz')['qpos']
tip = m.body('g1_a/wuji_right/r_thumb_tip_sensor_frame').id
js = [j for j in range(m.njnt) if m.joint(j).name.startswith('g1_a/wuji_right/r_thumb_')]
qa = np.array([m.jnt_qposadr[j] for j in js], dtype=int)
va = np.array([m.jnt_dofadr[j] for j in js], dtype=int)
results = []
for y in (-.385, -.39, -.395, -.40, -.405):
    for z in (.71, .715, .72, .725):
        d.qpos[:] = base
        goal = np.array([-.41, y, z])
        for _ in range(200):
            mujoco.mj_forward(m, d)
            error = goal-d.xpos[tip]
            if np.linalg.norm(error) < .001:
                break
            jac = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jac, None, d.xpos[tip], tip)
            J = jac[:, va]
            delta = J.T @ np.linalg.solve(J@J.T+.02**2*np.eye(3), error)
            d.qpos[qa] += np.clip(delta, -.04, .04)
            for j, adr in zip(js, qa):
                low, high = m.jnt_range[j]
                d.qpos[adr] = np.clip(d.qpos[adr], low+.03, high-.03)
        mujoco.mj_forward(m, d)
        bad = []
        pad = []
        for i, c in enumerate(d.contact):
            if c.geom1 < 0 or c.geom2 < 0:
                continue
            names = [m.geom(g).name for g in (c.geom1,c.geom2)]
            bodies = [m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
            if 'floor' in names or not any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
                continue
            if c.dist < -.003:
                bad.append((bodies, float(c.dist)))
            if 'dressing_pad_contact' in names:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, wrench)
                pad.append({'bodies': bodies, 'distance_m': float(c.dist),
                            'force_n': float(wrench[0])})
        results.append({'goal_m': goal.tolist(), 'thumb_tip_m': d.xpos[tip].tolist(),
                        'tip_error_m': float(np.linalg.norm(d.xpos[tip]-goal)),
                        'thumb_qrad': d.qpos[qa].tolist(),
                        'deep_contacts': bad, 'pad_contacts': pad})
report = {'scope': 'Static thumb-only IK scan from four-arm precontact seed; no dynamic grasp',
          'candidates': results}
(OUT/'pad_thumb_scan.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps([{'goal': x['goal_m'], 'error_mm': round(x['tip_error_m']*1000,1),
                   'deep': len(x['deep_contacts']), 'pad': len(x['pad_contacts'])}
                  for x in results]))
