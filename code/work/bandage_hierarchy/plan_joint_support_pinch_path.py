"""Static collision screen from held A-left support to A-right pinch candidate."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--pregrasp',action='store_true')
args=parser.parse_args()
model = mujoco.MjModel.from_xml_path(str(OUT / 'joint_support_pinch_variant.xml'))
data = mujoco.MjData(model)
base = np.load(OUT / 'dynamic_palm_support_rrt.npz')['qpos'][-1].copy()
goal_full = np.load(OUT / ('joint_support_pinch_pregrasp.npz' if args.pregrasp
                           else 'joint_support_pinch_static_candidate.npz'))['qpos']
joint_names = json.loads((OUT / 'joint_support_pinch_static_screen.json').read_text(encoding='utf8'))['transferred_right_joints']
joint_ids = [model.joint(name).id for name in joint_names]
qa = np.array([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
lo = np.array([model.jnt_range[j, 0] + .03 for j in joint_ids])
hi = np.array([model.jnt_range[j, 1] - .03 for j in joint_ids])
start = base[qa].copy()
goal = goal_full[qa].copy()
rng = np.random.default_rng(20261110)
checks = 0

def clear(q):
    global checks
    checks += 1
    data.qpos[:] = base
    data.qpos[qa] = q
    mujoco.mj_forward(model, data)
    for c in data.contact:
        if c.geom1 < 0 or c.geom2 < 0:
            continue
        geoms = (model.geom(c.geom1).name, model.geom(c.geom2).name)
        bodies = (model.body(model.geom_bodyid[c.geom1]).name,
                  model.body(model.geom_bodyid[c.geom2]).name)
        if args.pregrasp and 'dressing_pad_contact' in geoms and any(
                b.startswith('g1_a/wuji_right/') for b in bodies):
            return False
        if c.dist >= -.003:
            continue
        if 'floor' not in geoms and any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies):
            return False
    return True

def edge(a, b):
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / .04)))
    return all(clear((1 - u) * a + u * b) for u in np.linspace(0, 1, n + 1)[1:])

def extend(nodes, parents, target, step=.18):
    near = int(np.argmin(np.sum((np.asarray(nodes) - target)**2, axis=1)))
    delta = target - nodes[near]
    dist = float(np.linalg.norm(delta))
    candidate = nodes[near] + delta * min(1., step / max(dist, 1e-12))
    if not edge(nodes[near], candidate):
        return None
    nodes.append(candidate)
    parents.append(near)
    return len(nodes) - 1

def trace(nodes, parents, index):
    out = []
    while index >= 0:
        out.append(nodes[index])
        index = parents[index]
    return out[::-1]

endpoints_valid = clear(start) and clear(goal)
found = None
iterations = 0
if endpoints_valid:
    trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]
    for iteration in range(2000):
        iterations = iteration + 1
        if rng.random() < .2:
            sample = trees[1][0][0].copy()
        else:
            u = rng.random()
            sample = np.clip((1-u)*start + u*goal + rng.normal(0, .28, len(start)), lo, hi)
        active = iteration % 2
        other = 1 - active
        ni = extend(*trees[active], sample)
        if ni is None:
            continue
        target = trees[active][0][ni]
        for _ in range(30):
            nj = extend(*trees[other], target)
            if nj is None:
                break
            if np.linalg.norm(trees[other][0][nj] - target) < 1e-7:
                found = (trace(*trees[0], ni) + trace(*trees[1], nj)[::-1]
                         if active == 0 else trace(*trees[0], nj) + trace(*trees[1], ni)[::-1])
                break
        if found is not None:
            break
if found is not None:
    raw_waypoints = len(found)
    for _ in range(120):
        if len(found) < 3:
            break
        i, j = sorted(rng.choice(len(found), size=2, replace=False))
        if j > i + 1 and edge(found[i], found[j]):
            found = found[:i+1] + found[j:]
    recheck = all(edge(a, b) for a, b in zip(found, found[1:]))
    full = np.repeat(base[None, :], len(found), axis=0)
    for row, q in zip(full, found):
        row[qa] = q
    stem='joint_support_pregrasp_rrt' if args.pregrasp else 'joint_support_pinch_rrt'
    np.savez_compressed(OUT / f'{stem}.npz', qpos=full)
else:
    raw_waypoints = 0
    recheck = False
report = {'scope': 'Static right-hand approach while A-left support pose is held; no dynamic WBC or grasp proof',
          'goal_kind':'contact_free_pregrasp' if args.pregrasp else 'pinch_contact',
          'random_seed': 20261110, 'endpoints_valid': endpoints_valid,
          'found_path': found is not None, 'iterations': iterations,
          'path_waypoints': len(found) if found is not None else 0,
          'raw_waypoints': raw_waypoints, 'dense_recheck_passed': recheck,
          'collision_checks': checks}
(OUT / ('joint_support_pregrasp_rrt_report.json' if args.pregrasp else
        'joint_support_pinch_rrt_report.json')).write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report))
