"""Screen B-right Wuji hand postures at the fixed tab; static geometry only."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--sigma-rad', type=float, default=.075)
parser.add_argument('--samples', type=int, default=400)
parser.add_argument('--flex-contact-radius-mm', type=float, default=1.)
parser.add_argument('--middle-flex-offset-rad', type=float, default=0.)
parser.add_argument('--wrist-sigma-rad', type=float, default=0.)
parser.add_argument('--contact-scope', choices=('tip','digit'), default='digit')
args = parser.parse_args()
if not 0 < args.sigma_rad <= .4 or not 1 <= args.samples <= 10000:
    parser.error('sigma-rad must be in (0,0.4] and samples in [1,10000]')
if not 1 <= args.flex_contact_radius_mm <= 3:
    parser.error('flex-contact-radius-mm must be in [1,3]')
if not -.5 <= args.middle_flex_offset_rad <= .5:
    parser.error('middle-flex-offset-rad must be in [-0.5,0.5]')
if not 0 <= args.wrist_sigma_rad <= .2:
    parser.error('wrist-sigma-rad must be in [0,0.2]')
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
SCENE = OUT / 'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'
PATH = OUT / 'joint_b_pinchtab_rrt_path.npz'
m = mujoco.MjModel.from_xml_path(str(SCENE))
m.flex_radius[:] = args.flex_contact_radius_mm / 1000
d = mujoco.MjData(m)
q_ref = d.qpos.copy()
q_ref[:167] = np.load(PATH)['qpos'][-1, :167]
for joint_name in ('g1_b/wuji_right/r_middle_finger_mcp_flex',
                   'g1_b/wuji_right/r_middle_finger_pip',
                   'g1_b/wuji_right/r_middle_finger_dip'):
    joint = m.joint(joint_name)
    adr = int(joint.qposadr[0])
    q_ref[adr] = np.clip(q_ref[adr] + args.middle_flex_offset_rad,
                         m.jnt_range[joint.id, 0] + .01, m.jnt_range[joint.id, 1] - .01)
seed = np.load(OUT / 'joint_b_flex_a_thumb_side_contact_candidates.npz')['qpos'][0]
for j in range(m.njnt):
    if m.joint(j).name.startswith(('g1_a/wuji_right/r_thumb_', 'g1_a/wuji_right/r_index_finger_')):
        adr = int(m.jnt_qposadr[j])
        q_ref[adr] = seed[adr]
names = [m.joint(j).name for j in range(m.njnt)
         if m.joint(j).name.startswith(('g1_b/wuji_right/r_thumb_',
                                       'g1_b/wuji_right/r_index_finger_'))]
jids = np.array([m.joint(name).id for name in names])
qa = m.jnt_qposadr[jids]
low, high = m.jnt_range[jids, 0] + .01, m.jnt_range[jids, 1] - .01
wrist_names = [f'g1_b/right_wrist_{axis}_joint' for axis in ('roll','pitch','yaw')]
wrist_jids = np.array([m.joint(name).id for name in wrist_names])
wrist_qa = m.jnt_qposadr[wrist_jids]
tips = {'thumb': 'g1_b/wuji_right/r_thumb_tip_sensor_frame',
        'index': 'g1_b/wuji_right/r_index_finger_tip_sensor_frame'}
rng = np.random.default_rng(20261127)
rows = []
for trial in range(args.samples + 1):
    offset = np.zeros(len(qa)) if trial == 0 else rng.normal(0., args.sigma_rad, len(qa))
    d.qpos[:] = q_ref
    d.qpos[qa] = np.clip(q_ref[qa] + offset, low, high)
    if trial and args.wrist_sigma_rad:
        d.qpos[wrist_qa] = np.clip(q_ref[wrist_qa] + rng.normal(0., args.wrist_sigma_rad, 3),
                                  m.jnt_range[wrist_jids, 0] + .01,
                                  m.jnt_range[wrist_jids, 1] - .01)
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
        for digit, tip in tips.items():
            contacted=(tip in bodies if args.contact_scope=='tip' else
                       any(b.startswith(f'g1_b/wuji_right/r_{digit}_') for b in bodies))
            if contacted:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, wrench)
                forces[digit] += max(0., float(wrench[0]))
                normals[digit].append(c.frame[:3].copy())
    opposed = any(float(a @ b) < -.5 for a in normals['thumb'] for b in normals['index'])
    row = {'trial': trial, 'min_tip_normal_force_n': min(forces.values()),
           'tip_normal_force_n': forces, 'opposed': opposed,
           'max_b_nonfloor_penetration_m': max_pen,
           'max_hand_joint_offset_rad': float(np.max(np.abs(d.qpos[qa] - q_ref[qa]))),
           'joint_targets_rad': d.qpos[qa].tolist()}
    row['wrist_targets_rad'] = d.qpos[wrist_qa].tolist()
    row['static_screen_pass'] = bool(opposed and max_pen <= .003 and min(forces.values()) >= .5)
    rows.append(row)
safe_opposed = [r for r in rows if r['opposed'] and r['max_b_nonfloor_penetration_m'] <= .003]
loaded_opposed = [r for r in rows if r['opposed'] and r['min_tip_normal_force_n'] >= .5]
best_safe = max(safe_opposed, key=lambda r: r['min_tip_normal_force_n'], default=None)
least_penetrating_loaded = min(loaded_opposed, key=lambda r: r['max_b_nonfloor_penetration_m'], default=None)
rows.sort(key=lambda r: (not r['static_screen_pass'], not r['opposed'],
                         r['max_b_nonfloor_penetration_m'] > .003,
                         -r['min_tip_normal_force_n'], r['max_b_nonfloor_penetration_m']))
report = {'scope': 'Fixed-state posture screen only; contact force is a MuJoCo instantaneous estimate, not a stable grasp or robot motion',
          'scene': SCENE.name, 'samples': len(rows), 'joint_names': names,
          'sigma_rad': args.sigma_rad,
          'flex_contact_radius_mm': args.flex_contact_radius_mm,
          'middle_flex_offset_rad': args.middle_flex_offset_rad,
          'wrist_sigma_rad': args.wrist_sigma_rad,
          'contact_scope':args.contact_scope,
          'static_screen_pass_count': sum(r['static_screen_pass'] for r in rows),
          'safe_opposed_count': len(safe_opposed),
          'best_safe_opposed': best_safe,
          'loaded_opposed_count': len(loaded_opposed),
          'least_penetrating_loaded_opposed': least_penetrating_loaded,
          'baseline': next(r for r in rows if r['trial'] == 0), 'best': rows[0],
          'top_candidates': rows[:20]}
target = OUT / (f'joint_b_right_loaded_pinch_static_screen_sigma{args.sigma_rad:g}'
                f'_radius{args.flex_contact_radius_mm:g}mm'
                f'_middleoffset{args.middle_flex_offset_rad:g}rad'
                f'_wristsigma{args.wrist_sigma_rad:g}_scope{args.contact_scope}.json')
target.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'top_candidates'}))
