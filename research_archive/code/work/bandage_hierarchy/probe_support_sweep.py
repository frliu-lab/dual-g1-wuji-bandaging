"""Read-only geometric audit of the current A-left support against elbow rotation."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
m = mujoco.MjModel.from_xml_path(str(OUT / 'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'))
d = mujoco.MjData(m)
waypoints = np.load(OUT / 'joint_b_pinchtab_rrt_path.npz')['qpos']
d.qpos[:167] = waypoints[-1, :167]
elbow = m.joint('training_elbow_hinge').qposadr[0]
forearm = m.geom('manikin_forearm').id
wrist = m.body('g1_a/wuji_left/l_wrist').id
palms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == wrist and m.geom_contype[g]]
if not palms:
    raise RuntimeError('No collidable support palm geometry')
rows = []
for angle in (0.25, 0.28, 0.30, 0.32, 0.34, 0.35):
    d.qpos[elbow] = angle
    mujoco.mj_forward(m, d)
    candidates = []
    for palm in palms:
        points = np.zeros(6)
        dist = float(mujoco.mj_geomDistance(m, d, palm, forearm, 1., points))
        local = d.xmat[m.geom_bodyid[forearm]].reshape(3, 3).T @ (points[3:] - d.xpos[m.geom_bodyid[forearm]])
        candidates.append((dist, palm, points, local))
    dist, palm, points, local = min(candidates, key=lambda x: abs(x[0]))
    rows.append({'elbow_rad': angle, 'palm_forearm_distance_m': dist,
                 'palm_geom': m.geom(palm).name, 'forearm_local_nearest_m': local.tolist(),
                 'closest_points_world_m': points.tolist()})
print(json.dumps({'scope': 'Static support geometry sweep; no dynamics or load claim', 'rows': rows}, indent=2))
