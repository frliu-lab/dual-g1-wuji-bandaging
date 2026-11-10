"""Compare a moving forearm with and without the provisional A-left contact path."""
import json
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--candidate-path',action='store_true')
parser.add_argument('--reach-fraction',type=float,default=1.)
parser.add_argument('--segment-s',type=float,default=None)
parser.add_argument('--index-extra',type=float,default=0.)
args=parser.parse_args()
sys.path.insert(0, str(ROOT / 'work/wuji_multiagent'))
from joint_wbc import JointWBC

old = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
path_old = np.load(OUT / 'forearm_support_rrt.npz')['qpos']
scene = ET.parse(OUT / 'dynamic_forearm_variant.xml').getroot()
scene.find(".//body[@name='g1_a/pelvis']").set('pos', '-.3 0 .8' if args.candidate_path else '-.58 0 .8')
model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))

def remap(q):
    result = model.qpos0.copy()
    for j in range(old.njnt):
        name = old.joint(j).name
        new_id = model.joint(name).id
        n = 7 if old.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else (4 if old.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1)
        result[model.jnt_qposadr[new_id]:model.jnt_qposadr[new_id]+n] = q[old.jnt_qposadr[j]:old.jnt_qposadr[j]+n]
    return result

path = (np.load(OUT/'dynamic_palm_support_rrt.npz')['qpos'] if args.candidate_path
        else np.array([remap(q) for q in path_old]))
if args.candidate_path:
    path[-1]=path[0]+args.reach_fraction*(path[-1]-path[0])
    if args.index_extra:
        for j in range(model.njnt):
            name=model.joint(j).name
            if not name.startswith('g1_a/wuji_left/l_index_') or 'abd' in name:
                continue
            if not any(name.endswith(s) for s in ('_mcp_flex','_pip','_dip')):
                continue
            a=int(model.jnt_qposadr[j]);lo,hi=model.jnt_range[j]
            path[:,a]=np.clip(path[:,a]+args.index_extra,lo+.03,hi-.03)
palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
def contact_filter(m, c):
    return any(m.body(m.geom_bodyid[int(g)]).name.startswith(('g1_a/', 'g1_b/'))
               for g in c.geom if g >= 0)

jid = model.joint('training_elbow_hinge').id
qadr = int(model.jnt_qposadr[jid])
vadr = int(model.jnt_dofadr[jid])
geom_id = model.geom('manikin_forearm').id
hz = 50
interval = round(1 / hz / model.opt.timestep)
duration = 3.8
segment_s=args.segment_s if args.segment_s is not None else (1.2 if args.candidate_path else .6)
out = {}
for condition in ('no_contact', 'palm_candidate_path' if args.candidate_path else 'provisional_pinky_path'):
    data = mujoco.MjData(model)
    data.qpos[:] = path[0]
    mujoco.mj_forward(model, data)
    wbc = JointWBC(model, data.qpos.copy(), palm_names=palms, contact_filter=contact_filter)
    trace = []
    errors = []
    impulse_by_body = {}
    foot_support = 0
    deep = 0
    peak_force = 0.
    initial = float(data.qpos[qadr])
    for k in range(round(duration * hz)):
        if condition != 'no_contact':
            segment_steps=round(segment_s*hz)
            segment = min(k // segment_steps + 1, len(path) - 1)
            u = min(1., (k % segment_steps + 1) / segment_steps) if k < (len(path)-1)*segment_steps else 1.
            u = u*u*(3-2*u)
            wbc.qref[:] = (1-u)*path[segment-1] + u*path[segment]
        else:
            wbc.qref[:] = path[0]
        try:
            data.ctrl[:], _, _ = wbc.solve(data)
            for _ in range(interval):
                mujoco.mj_step(model, data)
                supported = set()
                force = 0.
                for ci, c in enumerate(data.contact):
                    if c.geom1 < 0 or c.geom2 < 0:
                        continue
                    names = [model.geom(g).name for g in (c.geom1, c.geom2)]
                    bodies = [model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
                    if 'floor' in names:
                        for label in ('g1_a', 'g1_b'):
                            if any(b.startswith(label + '/') for b in bodies):
                                supported.add(label)
                    elif any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies):
                        if c.dist < -.003:
                            deep += 1
                        if 'manikin_forearm' in names:
                            hand = next((b for b in bodies if b.startswith('g1_a/wuji_left/')), None)
                            if hand:
                                wrench = np.zeros(6)
                                mujoco.mj_contactForce(model, data, ci, wrench)
                                fn = max(0., float(wrench[0]))
                                force += fn
                                impulse_by_body[hand] = impulse_by_body.get(hand, 0.) + fn*model.opt.timestep
                foot_support += int(supported == {'g1_a', 'g1_b'})
                peak_force = max(peak_force, force)
            trace.append({'time_s': float(data.time), 'elbow_rad': float(data.qpos[qadr]),
                          'elbow_speed_rad_s': float(data.qvel[vadr]),
                          'forearm_center_m': data.geom_xpos[geom_id].tolist(),
                          'a_left_forearm_force_n': force})
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
            break
    final = trace[-25:]
    if not final:
        raise RuntimeError(f'{condition} produced no valid WBC samples: {errors}')
    out[condition] = {'simulated_time_s': float(data.time), 'initial_elbow_rad': initial,
        'final_elbow_rad': float(data.qpos[qadr]),
        'final_half_second_mean_abs_elbow_rad': float(np.mean([abs(x['elbow_rad']) for x in final])),
        'final_half_second_mean_force_n': float(np.mean([x['a_left_forearm_force_n'] for x in final])),
        'final_half_second_contact_fraction': float(np.mean([x['a_left_forearm_force_n'] >= .05 for x in final])),
        'peak_a_left_forearm_force_n': peak_force,
        'a_left_impulse_by_hand_body_ns': impulse_by_body,
        'foot_support_fraction': foot_support/max(1, len(trace)*interval),
        'deep_nonfloor_robot_contact_count': deep,
        'finite_state': bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        'errors': errors, 'trace': trace}
report = {'scope': ('Dynamic training-forearm WBC counterfactual for a static palm candidate; no validity claim' if args.candidate_path else
                    'Dynamic training-forearm WBC counterfactual; provisional path is pinky-only and rejected as humanlike support'),
          'forearm_mass_kg': .8, 'reach_fraction':args.reach_fraction,'segment_s':segment_s,'index_extra_rad':args.index_extra,'conditions': out}
name='dynamic_forearm_support_comparison.json'
if args.candidate_path:
    name='dynamic_palm_support_wbc_comparison'
    if args.reach_fraction!=1.:name+='_reach'+str(args.reach_fraction).replace('.','p')
    if args.segment_s is not None:name+='_segment'+str(segment_s).replace('.','p')
    if args.index_extra:name+='_index'+str(args.index_extra).replace('.','p')
    name+='.json'
(OUT/name).write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps({k: {j:v for j,v in s.items() if j!='trace'} for k,s in out.items()}, indent=2))
