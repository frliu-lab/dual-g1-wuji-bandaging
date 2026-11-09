"""Search B-right precontact waypoints with collision-aware path screening."""
import json
from itertools import product
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
model = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
data = mujoco.MjData(model)
seed = np.load(OUT / 'reach_ik_seed.npz')
start_q = seed['path_qpos'][3].copy()
body_id = model.body('g1_b/wuji_right/r_index_finger_tip_sensor_frame').id
joints = [j for j in range(model.njnt)
          if model.joint(j).name.startswith('g1_b/right_')
          and any(s in model.joint(j).name for s in ('shoulder', 'elbow', 'wrist'))
          and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
qadr = np.array([model.jnt_qposadr[j] for j in joints], dtype=int)
vadr = np.array([model.jnt_dofadr[j] for j in joints], dtype=int)
goal = np.array([.33, -.10, .89])


def solve(target):
    for _ in range(200):
        mujoco.mj_forward(model, data)
        error = target-data.xpos[body_id]
        if np.linalg.norm(error) < .01:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, data.xpos[body_id], body_id)
        J = jacp[:, vadr]
        dq = J.T@np.linalg.solve(J@J.T+.03**2*np.eye(3), error)
        data.qpos[qadr] += np.clip(dq, -.05, .05)
        for j, adr in zip(joints, qadr):
            if model.jnt_limited[j]:
                low, high = model.jnt_range[j]
                data.qpos[adr] = np.clip(data.qpos[adr], low+.03, high-.03)
    mujoco.mj_forward(model, data)


def depth():
    worst = 0.
    for c in data.contact:
        if min(c.geom1, c.geom2) < 0 or c.dist >= -.003:
            continue
        g = (int(c.geom1), int(c.geom2))
        names = [model.geom(i).name for i in g]
        bodies = [model.body(model.geom_bodyid[i]).name for i in g]
        if 'floor' not in names and any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies):
            worst = max(worst, float(-c.dist))
    return worst


data.qpos[:] = start_q
solve(goal)
direct_end_q = data.qpos.copy()
direct_goal_error = float(np.linalg.norm(data.xpos[body_id]-goal))
direct_end_depth = depth()
direct_worst = 0.
direct_collision_samples = 0
for u in np.linspace(0, 1, 101):
    data.qpos[:] = (1-u)*start_q+u*direct_end_q
    mujoco.mj_forward(model, data)
    penetration = depth()
    direct_worst = max(direct_worst, penetration)
    direct_collision_samples += penetration > .003
direct = {'goal_error_m': direct_goal_error, 'end_penetration_m': direct_end_depth,
          'path_worst_penetration_m': direct_worst,
          'collision_samples_of_101': direct_collision_samples}

rows = []
for waypoint in product((.36, .40, .44), (.12, .20, .28), (.88, .95, 1.02)):
    data.qpos[:] = start_q
    solve(np.array(waypoint))
    mid_q = data.qpos.copy()
    mid_error = float(np.linalg.norm(data.xpos[body_id]-waypoint))
    solve(goal)
    end_q = data.qpos.copy()
    end_error = float(np.linalg.norm(data.xpos[body_id]-goal))
    end_depth = depth()
    worst = 0.
    collision_samples = 0
    for a, b in ((start_q, mid_q), (mid_q, end_q)):
        for u in np.linspace(0, 1, 51):
            data.qpos[:] = (1-u)*a+u*b
            mujoco.mj_forward(model, data)
            penetration = depth()
            worst = max(worst, penetration)
            collision_samples += penetration > .003
    rows.append({'waypoint_m': waypoint, 'waypoint_error_m': mid_error,
                 'goal_error_m': end_error, 'end_penetration_m': end_depth,
                 'path_worst_penetration_m': worst,
                 'collision_samples_of_102': collision_samples})
rows.sort(key=lambda r: (r['goal_error_m'] >= .01, r['end_penetration_m'] > .003,
                         r['collision_samples_of_102'], r['path_worst_penetration_m']))
(OUT / 'guide_waypoint_search.json').write_text(json.dumps({'scope': 'Direct reach and 27 geometric waypoint candidates; no time-parameterized motion',
                                                           'direct': direct, 'candidates': rows}, indent=2), encoding='utf-8')
print(json.dumps({'direct': direct, 'best_waypoints': rows[:3]}))
