"""Static A-left arm search in the full dual-G1 scene; no executed entry path."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
m = mujoco.MjModel.from_xml_path(str(OUT / 'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'))
d = mujoco.MjData(m)
waypoints = np.load(OUT / 'joint_b_pinchtab_rrt_path.npz')['qpos']
base = m.qpos0.copy()
base[:167] = waypoints[-1, :167]
elbow = m.joint('training_elbow_hinge').qposadr[0]
forearm = m.geom('manikin_forearm').id
wrist = m.body('g1_a/wuji_left/l_wrist').id
palms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == wrist and m.geom_contype[g]]
names = [m.joint(i).name for i in range(m.njnt) if m.joint(i).name.startswith('g1_a/left_')
         and any(s in m.joint(i).name for s in ('shoulder', 'elbow', 'wrist'))
         and m.joint(i).type == mujoco.mjtJoint.mjJNT_HINGE]
qadr = np.array([m.joint(n).qposadr[0] for n in names])
bounds = []
for n in names:
    j = m.joint(n)
    lo, hi = m.jnt_range[j.id]
    center = base[j.qposadr[0]]
    bounds.append((max(lo + .03, center - .65), min(hi - .03, center + .65)))

def evaluate(x, detail=False):
    d.qpos[:] = base
    d.qpos[qadr] = x
    d.qpos[elbow] = .25
    mujoco.mj_forward(m, d)
    candidates = []
    for palm in palms:
        points = np.zeros(6)
        dist = float(mujoco.mj_geomDistance(m, d, palm, forearm, 1., points))
        local = d.xmat[m.geom_bodyid[forearm]].reshape(3, 3).T @ (points[3:] - d.xpos[m.geom_bodyid[forearm]])
        candidates.append((abs(dist + .0005) + abs(local[0] - .05), dist, local, palm))
    _, dist, local, palm = min(candidates)
    deep = []
    for c in d.contact:
        if c.dist >= -.003 or any(g < 0 for g in c.geom):
            continue
        geoms = (int(c.geom1), int(c.geom2))
        bodies = [m.body(m.geom_bodyid[g]).name for g in geoms]
        if any(b.startswith('g1_a/') for b in bodies) and 'floor' not in [m.geom(g).name for g in geoms]:
            if forearm in geoms and palm in geoms:
                continue
            deep.append(-float(c.dist) - .003)
    score = 300*abs(dist + .0005) + 30*abs(float(local[0]) - .05)
    score += 500*sum(deep) + .05*float(np.linalg.norm(x - base[qadr]))
    if detail:
        return {'score': score, 'palm_distance_m': dist, 'forearm_local_contact_m': local.tolist(),
                'deep_penetration_excess_m': deep, 'joint_names': names, 'joint_positions_rad': x.tolist()}
    return score

opt = differential_evolution(evaluate, bounds, seed=20261126, maxiter=60, popsize=9,
                             tol=.003, polish=True, updating='immediate')
result = evaluate(opt.x, True)
eligible = (abs(result['palm_distance_m']) <= .003 and
            .025 <= result['forearm_local_contact_m'][0] <= .08 and
            not result['deep_penetration_excess_m'])
report = {'scope': 'Static full-scene near-elbow support pose search; no path or force claim',
          'baseline': evaluate(base[qadr], True), 'candidate': result,
          'optimizer_success': bool(opt.success), 'eligible_static': bool(eligible)}
(OUT / 'joint_a_left_near_elbow_support_search.json').write_text(json.dumps(report, indent=2), encoding='utf8')
if eligible:
    d.qpos[:] = base
    d.qpos[qadr] = opt.x
    np.savez_compressed(OUT / 'joint_a_left_near_elbow_support_candidate.npz', qpos=d.qpos, joint_names=names)
print(json.dumps({'eligible_static': eligible, 'candidate': result, 'baseline': report['baseline']}))
