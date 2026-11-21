"""Offline search for a tip-first pinch pose on the diagnostic post."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser = argparse.ArgumentParser()
parser.add_argument('--stem', default='joint_dressing_pilot_1132321911a2cbf4')
parser.add_argument('--frame', type=int, default=124)
parser.add_argument('--output', default='joint_post_tip_first_ik_search.json')
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(str(OUT / 'joint_spool_grasp_dressing_pinch_post_variant.xml'))
trajectory = np.load(OUT / f'{args.stem}_qpos.npz')['qpos']
q0 = trajectory[args.frame].copy()
data = mujoco.MjData(model)
data.qpos[:] = q0
mujoco.mj_forward(model, data)
post = model.geom('dressing_pull_tab').id
center = data.geom_xpos[post].copy()
rotation = data.geom_xmat[post].reshape(3, 3).copy()
body = {name: model.body('g1_a/wuji_right/r_' + name).id for name in (
    'thumb_tip_sensor_frame', 'index_finger_tip_sensor_frame',
    'index_finger_middle', 'index_finger_proximal_abd', 'wuji_right/r_wrist')
    if name != 'wuji_right/r_wrist'}
body['wrist'] = model.body('g1_a/wuji_right/r_wrist').id
ids = [j for j in range(model.njnt) if model.joint(j).name.startswith('g1_a/right_')
       and any(word in model.joint(j).name for word in ('shoulder', 'elbow', 'wrist'))]
for prefix in ('r_thumb_', 'r_index_finger_'):
    ids += [j for j in range(model.njnt)
            if model.joint(j).name.startswith('g1_a/wuji_right/' + prefix)]
qa = np.array([model.jnt_qposadr[j] for j in ids])
start = q0[qa].copy()
lo = model.jnt_range[ids, 0] + 1e-5
hi = model.jnt_range[ids, 1] - 1e-5
wrist_start = data.xpos[body['wrist']].copy()
results = []
for side_sign in (1,):
    for z_offset in (-.004, .002):
        for thumb_offset in (.019, .021):
          for index_offset in (.012, .014, .016, .018, .020):
            middle_clearance = .030
            targets = {
                'thumb': center + side_sign * rotation[:, 1] * thumb_offset + rotation[:, 2] * z_offset,
                'index': center - side_sign * rotation[:, 1] * index_offset + rotation[:, 2] * z_offset,
            }

            def residual(x):
                data.qpos[qa] = x
                mujoco.mj_forward(model, data)
                middle_r = np.linalg.norm(data.xpos[body['index_finger_middle'], :2] - center[:2])
                proximal_r = np.linalg.norm(data.xpos[body['index_finger_proximal_abd'], :2] - center[:2])
                penetration = {'thumb': 0., 'index': 0., 'other': 0.}
                for contact in data.contact:
                    if post not in (contact.geom1, contact.geom2):
                        continue
                    other_geom = contact.geom2 if contact.geom1 == post else contact.geom1
                    other_body = model.body(model.geom_bodyid[other_geom]).name
                    if not other_body.startswith('g1_a/wuji_right/'):
                        continue
                    group = ('thumb' if '/r_thumb_' in other_body else
                             'index' if '/r_index_finger_' in other_body else 'other')
                    penetration[group] = max(penetration[group], -float(contact.dist))
                return np.r_[5 * (data.xpos[body['thumb_tip_sensor_frame']] - targets['thumb']),
                             5 * (data.xpos[body['index_finger_tip_sensor_frame']] - targets['index']),
                             2 * max(0., middle_clearance - middle_r),
                             2 * max(0., .030 - proximal_r),
                             10 * max(0., penetration['thumb'] - .0007),
                             10 * max(0., penetration['index'] - .0007),
                             10 * max(0., penetration['other'] - .0007),
                             .05 * (data.xpos[body['wrist']] - wrist_start),
                             .001 * (x[:7] - start[:7])]

            fit = least_squares(residual, np.clip(start, lo, hi), bounds=(lo, hi), max_nfev=100)
            residual(fit.x)
            thumb_error = float(np.linalg.norm(data.xpos[body['thumb_tip_sensor_frame']] - targets['thumb']))
            index_error = float(np.linalg.norm(data.xpos[body['index_finger_tip_sensor_frame']] - targets['index']))
            middle_r = float(np.linalg.norm(data.xpos[body['index_finger_middle'], :2] - center[:2]))
            proximal_r = float(np.linalg.norm(data.xpos[body['index_finger_proximal_abd'], :2] - center[:2]))
            post_contacts = [c for c in data.contact if post in (c.geom1, c.geom2)
                             and any(model.body(model.geom_bodyid[g]).name.startswith('g1_a/wuji_right/')
                                     for g in (c.geom1, c.geom2))]
            max_penetration = max((max(0., -float(c.dist)) for c in post_contacts), default=0.)
            contact_bodies = [model.body(model.geom_bodyid[c.geom2 if c.geom1 == post else c.geom1]).name
                              for c in post_contacts]
            results.append({'side_sign': side_sign, 'z_offset_m': z_offset,
                            'thumb_offset_m': thumb_offset, 'index_offset_m': index_offset,
                            'middle_clearance_target_m': middle_clearance,
                            'thumb_tip_error_m': thumb_error, 'index_tip_error_m': index_error,
                            'index_middle_radius_m': middle_r, 'index_proximal_radius_m': proximal_r,
                            'wrist_shift_m': float(np.linalg.norm(data.xpos[body['wrist']] - wrist_start)),
                            'max_post_penetration_m': max_penetration,
                            'post_contact_bodies': contact_bodies,
                            'arm_delta_norm_rad': float(np.linalg.norm(fit.x[:7] - start[:7])),
                            'qpos': fit.x.tolist()})

results.sort(key=lambda r: (r['thumb_tip_error_m'] + r['index_tip_error_m']
                            + max(0., .026-r['index_middle_radius_m'])
                            + max(0., .030-r['index_proximal_radius_m'])
                            + 4*max(0.,r['max_post_penetration_m']-.001)))
report = {'scope': 'Static IK only; no dynamic grip or collision pass',
          'source_stem': args.stem, 'frame': args.frame,
          'joint_names': [model.joint(j).name for j in ids], 'results': results}
(OUT / args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'output': args.output, 'best': {k: v for k, v in results[0].items() if k != 'qpos'},
                  'candidate_count': len(results)}))
