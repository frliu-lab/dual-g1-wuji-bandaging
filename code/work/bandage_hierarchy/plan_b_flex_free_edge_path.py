"""Collision-screened static path to the B-right loose-strip pinch candidate."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--pinchtab',action='store_true')
args=parser.parse_args()
SCENE = OUT / ('joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab_bulged.xml'
               if args.pinchtab else
               'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5.xml')
SOURCE = OUT / 'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json'
model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
seed = json.loads((OUT / 'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
base = data.qpos.copy()
base[:len(seed['samples'][-1]['qpos'])] = seed['samples'][-1]['qpos']
screen = json.loads(SOURCE.read_text())
names = screen['joint_names']
qa = np.array([model.joint(name).qposadr[0] for name in names])
start = base[qa].copy()
goal = np.array(screen['results'][0]['joint_targets_rad'])
if args.pinchtab:
    pregrasp=json.loads((OUT/'joint_b_pinchtab_safe_start_screen.json').read_text())
    assert pregrasp['joint_names']==names
    start=np.asarray(pregrasp['results'][0]['joint_targets_rad'])
    hand=json.loads((OUT/'joint_b_natural_pinch_workspace_screen.json').read_text())
    candidate=next(row for row in hand['results'] if row['base']=='reach' and row['trial']==4)
    lookup={name:i for i,name in enumerate(names)}
    for name,value in zip(hand['joint_names'],candidate['joint_targets_rad']):
        goal[lookup[name]]=value
joint_ids = [model.joint(name).id for name in names]
lo = np.array([model.jnt_range[i, 0] for i in joint_ids])
hi = np.array([model.jnt_range[i, 1] for i in joint_ids])
arm = np.array([name.startswith('g1_b/right_') for name in names])
rng = np.random.default_rng(20261122)
checks = 0
worst_checked = 0.0

def clear(q):
    global checks, worst_checked
    checks += 1
    data.qpos[:] = base
    data.qpos[qa] = q
    mujoco.mj_forward(model, data)
    worst = 0.0
    for c in data.contact:
        if c.dist >= 0:
            continue
        geoms = [int(g) for g in c.geom if g >= 0]
        bodies = [model.body(model.geom_bodyid[g]).name for g in geoms]
        if not any(name.startswith('g1_b/') for name in bodies):
            continue
        if 'floor' in [model.geom(g).name for g in geoms]:
            continue
        worst = max(worst, -float(c.dist))
    worst_checked = max(worst_checked, worst)
    return worst < .003

def edge(a, b, resolution=.015):
    n = max(1, int(np.ceil(np.max(np.abs(b-a)) / resolution)))
    return all(clear(a+(b-a)*u) for u in np.linspace(1/n, 1, n))

def extend(nodes, parents, target, step=.16):
    near = int(np.argmin(np.sum((np.asarray(nodes)-target)**2, axis=1)))
    delta = target-nodes[near]
    dist = float(np.linalg.norm(delta))
    q = nodes[near] + delta*min(1, step/max(dist, 1e-12))
    if not edge(nodes[near], q):
        return None
    nodes.append(q)
    parents.append(near)
    return len(nodes)-1

def trace(nodes, parents, idx):
    result = []
    while idx >= 0:
        result.append(nodes[idx])
        idx = parents[idx]
    return result[::-1]

start_ok, goal_ok = clear(start), clear(goal)
found = None
iterations = 0
if start_ok and goal_ok:
    trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]
    for k in range(4000):
        iterations = k+1
        u = rng.random()
        if u < .10:
            sample = goal
        elif u < .2:
            sample = start
        else:
            t = rng.random()
            noise = rng.normal(0, rng.choice([.08, .2, .4]), len(start))
            if rng.random() < .25:
                noise[arm] = 0
            elif rng.random() < .25:
                noise[~arm] = 0
            sample = np.clip(start+t*(goal-start)+noise, lo, hi)
        active = k % 2
        other = 1-active
        ni = extend(*trees[active], sample)
        if ni is None:
            continue
        target = trees[active][0][ni]
        for _ in range(50):
            nj = extend(*trees[other], target)
            if nj is None:
                break
            if np.linalg.norm(trees[other][0][nj]-target) < 1e-7:
                if active == 0:
                    found = trace(*trees[0], ni)+trace(*trees[1], nj)[::-1]
                else:
                    found = trace(*trees[0], nj)+trace(*trees[1], ni)[::-1]
                break
        if found is not None:
            break

raw = len(found) if found is not None else 0
if found is not None:
    for _ in range(200):
        if len(found) < 3:
            break
        i, j = sorted(rng.choice(len(found), 2, replace=False))
        if j > i+1 and edge(found[i], found[j]):
            found = found[:i+1]+found[j:]
    dense_ok = all(edge(a, b, .015) for a, b in zip(found, found[1:]))
    path = np.repeat(base[None, :], len(found), axis=0)
    path[:, qa] = found
    np.savez_compressed(OUT/('joint_b_pinchtab_rrt_path.npz' if args.pinchtab else
                             'joint_b_flex_free_edge_rrt_path.npz'), qpos=path, joint_names=names)
else:
    dense_ok = False

report = {
    'scope': 'Static B-right 25-joint path with flexible strip frozen; no dynamic grip or wrapping claim',
    'scene': SCENE.name, 'source_screen': SOURCE.name,
    'pinchtab':args.pinchtab,
    'start_valid': bool(start_ok), 'goal_valid': bool(goal_ok),
    'found_path': found is not None, 'dense_recheck_passed': bool(dense_ok),
    'iterations': iterations, 'collision_checks': checks,
    'raw_waypoints': raw, 'waypoints': len(found) if found is not None else 0,
    'max_penetration_seen_among_all_samples_m': worst_checked,
}
(OUT/('joint_b_pinchtab_rrt_report.json' if args.pinchtab else
      'joint_b_flex_free_edge_rrt_report.json')).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
