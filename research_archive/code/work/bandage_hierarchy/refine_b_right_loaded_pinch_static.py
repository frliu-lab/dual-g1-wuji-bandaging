"""Local static posture search around safe and loaded B-right pinch candidates."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--radius-mm',type=float,choices=(1.5,2.),default=2.)
args=parser.parse_args()
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
source = OUT / (f'joint_b_right_loaded_pinch_static_screen_sigma0.075_radius{args.radius_mm:g}mm_'
                'middleoffset0.5rad_wristsigma0.06_scopedigit.json')
prior = json.loads(source.read_text())
m = mujoco.MjModel.from_xml_path(str(OUT / prior['scene']))
m.flex_radius[:] = prior['flex_contact_radius_mm'] / 1000
d = mujoco.MjData(m)
q_ref = d.qpos.copy()
q_ref[:167] = np.load(OUT / 'joint_b_pinchtab_rrt_path.npz')['qpos'][-1, :167]
for name in ('g1_b/wuji_right/r_middle_finger_mcp_flex',
             'g1_b/wuji_right/r_middle_finger_pip',
             'g1_b/wuji_right/r_middle_finger_dip'):
    jid = m.joint(name).id
    q_ref[m.jnt_qposadr[jid]] += prior['middle_flex_offset_rad']
seed = np.load(OUT / 'joint_b_flex_a_thumb_side_contact_candidates.npz')['qpos'][0]
for j in range(m.njnt):
    if m.joint(j).name.startswith(('g1_a/wuji_right/r_thumb_',
                                   'g1_a/wuji_right/r_index_finger_')):
        adr = int(m.jnt_qposadr[j])
        q_ref[adr] = seed[adr]
hand_jids = np.array([m.joint(name).id for name in prior['joint_names']])
hand_qa = m.jnt_qposadr[hand_jids]
wrist_names = [f'g1_b/right_wrist_{axis}_joint' for axis in ('roll','pitch','yaw')]
wrist_jids = np.array([m.joint(name).id for name in wrist_names])
wrist_qa = m.jnt_qposadr[wrist_jids]
jids = np.r_[hand_jids, wrist_jids]
qa = np.r_[hand_qa, wrist_qa]
low, high = m.jnt_range[jids, 0] + .01, m.jnt_range[jids, 1] - .01
starts = [prior['best_safe_opposed'], prior['least_penetrating_loaded_opposed']]
assert all(x is not None for x in starts)
rng = np.random.default_rng(20261127)
rows = []
for start_id, start in enumerate(starts):
    center = np.r_[start['joint_targets_rad'], start['wrist_targets_rad']]
    for trial in range(6001):
        sigma = .01 if trial < 3000 else .025
        offset = np.zeros(len(qa)) if trial == 0 else rng.normal(0., sigma, len(qa))
        d.qpos[:] = q_ref
        d.qpos[qa] = np.clip(center + offset, low, high)
        d.qvel[:] = 0.
        mujoco.mj_forward(m, d)
        forces = {'thumb': 0., 'index': 0.}
        normals = {'thumb': [], 'index': []}
        max_pen = 0.
        for ci, c in enumerate(d.contact):
            geoms = [int(g) for g in c.geom if g >= 0]
            bodies = [m.body(m.geom_bodyid[g]).name for g in geoms]
            if c.dist < 0 and any(b.startswith('g1_b/') for b in bodies):
                if 'floor' not in [m.geom(g).name for g in geoms]:
                    max_pen = max(max_pen, -float(c.dist))
            if not any(f >= 0 for f in c.flex):
                continue
            for digit in forces:
                if any(b.startswith(f'g1_b/wuji_right/r_{digit}_') for b in bodies):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, wrench)
                    forces[digit] += max(0., float(wrench[0]))
                    normals[digit].append(c.frame[:3].copy())
        opposed = any(float(a @ b) < -.5 for a in normals['thumb'] for b in normals['index'])
        row = {'start_id': start_id, 'trial': trial, 'min_digit_force_n': min(forces.values()),
               'digit_forces_n': forces, 'opposed': opposed,
               'max_b_nonfloor_penetration_m': max_pen,
               'joint_targets_rad': d.qpos[qa].tolist()}
        row['static_screen_pass'] = bool(opposed and max_pen <= .003 and
                                         row['min_digit_force_n'] >= .5)
        rows.append(row)
safe = [r for r in rows if r['opposed'] and r['max_b_nonfloor_penetration_m'] <= .003]
loaded = [r for r in rows if r['opposed'] and r['min_digit_force_n'] >= .5]
passing = [r for r in rows if r['static_screen_pass']]
report = {'scope': 'Static local posture search around two candidates; no motion or stable grasp',
          'source': source.name, 'samples': len(rows), 'joint_names': prior['joint_names'] + wrist_names,
          'passing_count': len(passing), 'best_passing': min(passing, key=lambda r:r['max_b_nonfloor_penetration_m'],default=None),
          'top_passing_clearance_force': sorted(passing,key=lambda r:(
              -(r['min_digit_force_n']-200*max(0.,r['max_b_nonfloor_penetration_m']-.002))))[:30],
          'best_safe': max(safe,key=lambda r:r['min_digit_force_n'],default=None),
          'least_penetrating_loaded': min(loaded,key=lambda r:r['max_b_nonfloor_penetration_m'],default=None)}
target = OUT / f'joint_b_right_loaded_pinch_static_refine_radius{args.radius_mm:g}mm.json'
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('best_passing','best_safe','least_penetrating_loaded','top_passing_clearance_force')}))
print(json.dumps({k: None if report[k] is None else
                  {n:report[k][n] for n in ('min_digit_force_n','max_b_nonfloor_penetration_m')}
                  for k in ('best_passing','best_safe','least_penetrating_loaded')}))
