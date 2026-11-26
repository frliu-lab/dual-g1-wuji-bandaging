"""Static clearance probe for the middle finger replacing A-left palm contact."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
m = mujoco.MjModel.from_xml_path(str(OUT / 'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'))
d = mujoco.MjData(m)
seed = np.load(OUT / 'joint_a_left_near_elbow_support_candidate.npz')
base = seed['qpos'].copy()
forearm = m.geom('manikin_forearm').id
middle = [g for g in range(m.ngeom)
          if m.body(m.geom_bodyid[g]).name == 'g1_a/wuji_left/l_middle_finger_proximal'
          and m.geom_contype[g]]
palm = [g for g in range(m.ngeom)
        if m.body(m.geom_bodyid[g]).name == 'g1_a/wuji_left/l_wrist' and m.geom_contype[g]]
joint_names = [m.joint(j).name for j in range(m.njnt)
               if m.joint(j).name.startswith('g1_a/wuji_left/l_middle_finger_')]

def distances():
    mujoco.mj_forward(m, d)
    return {'middle_m': min(float(mujoco.mj_geomDistance(m, d, g, forearm, 1., np.zeros(6))) for g in middle),
            'palm_m': min(float(mujoco.mj_geomDistance(m, d, g, forearm, 1., np.zeros(6))) for g in palm)}

cases = []
for elbow_angle in (.25, .30, .35):
    d.qpos[:] = base
    d.qpos[m.joint('training_elbow_hinge').qposadr[0]] = elbow_angle
    cases.append({'elbow_rad': elbow_angle, 'joint': 'baseline', 'delta_rad': 0., **distances()})
    for name in joint_names:
        joint = m.joint(name)
        adr = joint.qposadr[0]
        q = float(d.qpos[adr])
        for delta in (-.3, -.15, -.1, -.05, .05, .1, .15, .3, .5, .8, 1.):
            lo, hi = m.jnt_range[joint.id]
            if not lo + .01 <= q + delta <= hi - .01:
                continue
            d.qpos[adr] = q + delta
            cases.append({'elbow_rad': elbow_angle, 'joint': name, 'delta_rad': delta, **distances()})
            d.qpos[adr] = q
report = {'scope': 'Static A-left middle-finger clearance perturbations; no dynamic or load claim', 'cases': cases}
(OUT / 'joint_a_left_middle_forearm_clearance_probe.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps({'baseline': [c for c in cases if c['joint'] == 'baseline'],
                  'best_at_0p3': sorted((c for c in cases if c['elbow_rad'] == .30),
                                        key=lambda c: c['middle_m'], reverse=True)[:8]}))
