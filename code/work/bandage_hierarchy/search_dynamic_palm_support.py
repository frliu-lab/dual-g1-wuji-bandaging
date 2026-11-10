"""Search a geometric A-left palm support candidate on the moving forearm.

This is a pose screen only. It deliberately does not declare a grasp or a safe path.
"""
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--root-x',type=float,default=-.58)
parser.add_argument('--pinky-extra',type=float,default=0.)
parser.add_argument('--elbow-angle',type=float,default=.25)
args=parser.parse_args()
scene = ET.parse(OUT/'dynamic_forearm_variant.xml').getroot()
scene.find(".//body[@name='g1_a/pelvis']").set('pos', f'{args.root_x} 0 .8')
model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
data = mujoco.MjData(model)
old = mujoco.MjModel.from_xml_path(str(OUT/'scene.xml'))
old_pose = np.load(OUT/'forearm_support_rrt.npz')['qpos'][-1]
base = model.qpos0.copy()
for j in range(old.njnt):
    name = old.joint(j).name
    k = model.joint(name).id
    n = 7 if old.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else (4 if old.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1)
    base[model.jnt_qposadr[k]:model.jnt_qposadr[k]+n] = old_pose[old.jnt_qposadr[j]:old.jnt_qposadr[j]+n]
elbow_id = model.joint('training_elbow_hinge').id
base[model.jnt_qposadr[elbow_id]] = args.elbow_angle
base[model.joint('g1_a/floating_base').qposadr[0]]=args.root_x
for j in range(model.njnt):
    name=model.joint(j).name
    if not name.startswith('g1_a/wuji_left/l_pinky_') or 'abd' in name or not args.pinky_extra:
        continue
    a=int(model.jnt_qposadr[j]);lo,hi=model.jnt_range[j]
    base[a]=np.clip(base[a]+args.pinky_extra,lo+.03,hi-.03)
names = [model.joint(j).name for j in range(model.njnt)
         if model.joint(j).name.startswith('g1_a/left_')
         and any(s in model.joint(j).name for s in ('shoulder','elbow','wrist'))
         and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
names += ['g1_a/wuji_left/l_pinky_mcp_flex','g1_a/wuji_left/l_pinky_mcp_abd']
qadr = np.array([model.joint(n).qposadr[0] for n in names], dtype=int)
joint_ids = [model.joint(n).id for n in names]
bounds = []
for j in joint_ids:
    lo, hi = model.jnt_range[j]
    center = base[model.jnt_qposadr[j]]
    bounds.append((max(lo+.04, center-1.0), min(hi-.04, center+1.0)))
forearm = model.geom('manikin_forearm').id
palm_body = model.body('g1_a/wuji_left/l_wrist').id
palm = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == palm_body and model.geom_contype[g])
if model.geom_contype[palm] == 0:
    raise RuntimeError('The Wuji palm is not a collision geometry')
closest = np.zeros(6)
checks = 0

def evaluate(x, detail=False):
    global checks
    checks += 1
    data.qpos[:] = base
    data.qpos[qadr] = x
    mujoco.mj_forward(model, data)
    dist = float(mujoco.mj_geomDistance(model, data, palm, forearm, 1., closest))
    deep = []
    palm_contact = []
    finger_contacts = []
    for c in data.contact:
        if c.geom1 < 0 or c.geom2 < 0:
            continue
        geoms = (c.geom1,c.geom2)
        names_contact = [model.geom(g).name for g in geoms]
        bodies = [model.body(model.geom_bodyid[g]).name for g in geoms]
        if 'floor' in names_contact or not any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
            continue
        if c.dist < -.003:
            deep.append({'distance_m':float(c.dist),'bodies':bodies,'geoms':names_contact})
        if forearm in geoms and any(b.startswith('g1_a/wuji_left/') for b in bodies):
            if palm in geoms:
                palm_contact.append(float(c.dist))
            else:
                finger_contacts.append((next(b for b in bodies if b.startswith('g1_a/wuji_left/')), float(c.dist)))
    # Seeking a shallow palm contact while excluding any deep nonfloor robot collision.
    objective = 400*abs(dist+.0005) + 80*sum(max(0.,-.003-v['distance_m']) for v in deep)
    objective += 20*max(0.,.02-closest[3])
    objective += .4*len(finger_contacts)+100*sum(max(0.,-v) for _,v in finger_contacts)
    objective += .08*float(np.linalg.norm(x-base[qadr]))
    if detail:
        return {'objective':objective,'palm_forearm_distance_m':dist,
                'palm_forearm_closest_points_m':closest.tolist(),
                'deep_nonfloor_contact_distances_m':deep,
                'palm_contact_distances_m':palm_contact,
                'finger_forearm_contacts':finger_contacts,
                'joint_names':names,'joint_positions_rad':x.tolist(),
                'elbow_angle_rad':float(data.qpos[model.jnt_qposadr[elbow_id]])}
    return objective

baseline = evaluate(base[qadr], True)
rng = 20261109
opt = differential_evolution(evaluate, bounds, seed=rng, maxiter=100, popsize=12,
                             tol=.002, polish=True, updating='immediate')
candidate = evaluate(opt.x, True)
eligible = bool(candidate['palm_contact_distances_m'] and
                not candidate['deep_nonfloor_contact_distances_m'] and
                not candidate['finger_forearm_contacts'] and
                candidate['palm_forearm_closest_points_m'][3] >= .02 and
                abs(candidate['palm_forearm_distance_m']) < .003)
report = {'scope':'Static dynamic-forearm palm geometry search at prescribed elbow angle; no force, motion path, or WBC claim',
          'random_seed':rng,'a_root_x_m':args.root_x,'pinky_extra_rad':args.pinky_extra,'target_elbow_rad':args.elbow_angle,'checks':checks,'optimizer_success':bool(opt.success),
          'baseline':baseline,'candidate':candidate,'eligible_palm_contact':eligible}
stem='dynamic_palm_support_search_root_'+str(args.root_x).replace('-','m').replace('.','p')
if args.pinky_extra:stem+='_pinky'+str(args.pinky_extra).replace('-','m').replace('.','p')
if args.elbow_angle!=.25:stem+='_elbow'+str(args.elbow_angle).replace('-','m').replace('.','p')
(OUT/f'{stem}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
if eligible:
    np.savez_compressed(OUT/f'{stem}_candidate.npz',qpos=data.qpos)
print(json.dumps({k:v for k,v in report.items() if k not in ('baseline','candidate')}|{'candidate':candidate}))
