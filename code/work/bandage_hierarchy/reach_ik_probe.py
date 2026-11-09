"""Four-arm position-only IK probe for a staged bandaging reach.

This is a geometric seed, not a validated motion or closed-loop rollout.
"""
import json
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
model = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
for robot in ('g1_a', 'g1_b'):
    foot_geoms = [i for i in range(model.ngeom)
                  if model.geom(i).name.startswith(robot+'/')
                  and 'official_foot_contact' in model.geom(i).name]
    clearance = min(float(data.geom_xpos[i, 2]-model.geom_size[i, 0]) for i in foot_geoms)
    joint = model.joint(f'{robot}/floating_base').id
    data.qpos[model.jnt_qposadr[joint]+2] -= clearance+.0002
mujoco.mj_forward(model, data)
initial_qpos = data.qpos.copy()

targets = {
    'g1_a/left': ('g1_a/wuji_left/l_index_finger_tip_sensor_frame', np.array([-.22, .18, .84]), 'support precontact approach below proximal forearm'),
    'g1_a/right': ('g1_a/wuji_right/r_index_finger_tip_sensor_frame', np.array([-.38, -.44, .72]), 'dressing pad approach'),
    'g1_b/left': ('g1_b/wuji_left/l_index_finger_tip_sensor_frame', np.array([.38, -.44, .78]), 'bandage spool approach'),
    'g1_b/right': ('g1_b/wuji_right/r_index_finger_tip_sensor_frame', np.array([.45, -.08, .89]), 'guide precontact staging outside forearm'),
}

def solve_to(body_id, joints, qadr, vadr, target):
    for _ in range(200):
        mujoco.mj_forward(model, data)
        error = target-data.xpos[body_id]
        if np.linalg.norm(error) < .01:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, data.xpos[body_id], body_id)
        J = jacp[:, vadr]
        dq = J.T @ np.linalg.solve(J@J.T+.03**2*np.eye(3), error)
        data.qpos[qadr] += np.clip(dq, -.05, .05)
        for j, adr in zip(joints, qadr):
            if model.jnt_limited[j]:
                low, high = model.jnt_range[j]
                data.qpos[adr] = np.clip(data.qpos[adr], low+.03, high-.03)
    mujoco.mj_forward(model, data)


results = {}
path_keyframes = [('initial', initial_qpos.copy())]
for role, (wrist, target, purpose) in targets.items():
    robot, side = role.split('/')
    joints = [j for j in range(model.njnt)
              if model.joint(j).name.startswith(f'{robot}/{side}_')
              and any(token in model.joint(j).name for token in ('shoulder', 'elbow', 'wrist'))
              and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr = np.array([model.jnt_qposadr[j] for j in joints], dtype=int)
    vadr = np.array([model.jnt_dofadr[j] for j in joints], dtype=int)
    body_id = model.body(wrist).id
    start = data.xpos[body_id].copy()
    solve_to(body_id, joints, qadr, vadr, target)
    path_keyframes.append((role, data.qpos.copy()))
    results[role] = {'purpose': purpose, 'wrist': wrist,
                     'start_m': start.tolist(), 'target_m': target.tolist(),
                     'achieved_m': data.xpos[body_id].tolist(),
                     'initial_distance_m': float(np.linalg.norm(start-target)),
                     'final_position_error_m': float(np.linalg.norm(data.xpos[body_id]-target)),
                     'arm_joint_count': len(joints),
                     'max_arm_joint_change_rad': float(np.max(np.abs(data.qpos[qadr]-initial_qpos[qadr])))}

deep_contacts = []
for c in data.contact:
    if c.geom1 < 0 or c.geom2 < 0 or c.dist >= -.003:
        continue
    names = (model.geom(c.geom1).name, model.geom(c.geom2).name)
    bodies = (model.body(model.geom_bodyid[c.geom1]).name,
              model.body(model.geom_bodyid[c.geom2]).name)
    if any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies) and 'floor' not in names:
        deep_contacts.append({'geoms': names, 'bodies': bodies, 'distance_m': float(c.dist)})
final_qpos = data.qpos.copy()
path_deepest = 0.
path_collision_samples = 0
path_collision_examples = []
for (from_name, start_q), (to_name, end_q) in zip(path_keyframes, path_keyframes[1:]):
    for u in np.linspace(0., 1., 101):
        data.qpos[:] = (1-u)*start_q+u*end_q
        mujoco.mj_forward(model, data)
        sample_deep = False
        for c in data.contact:
            if c.geom1 < 0 or c.geom2 < 0 or c.dist >= -.003:
                continue
            names = (model.geom(c.geom1).name, model.geom(c.geom2).name)
            bodies = (model.body(model.geom_bodyid[c.geom1]).name,
                      model.body(model.geom_bodyid[c.geom2]).name)
            if any(b.startswith(('g1_a/', 'g1_b/')) for b in bodies) and 'floor' not in names:
                sample_deep = True
                path_deepest = max(path_deepest, float(-c.dist))
                path_collision_examples.append({'stage': f'{from_name}->{to_name}', 'u': float(u),
                                                'bodies': bodies, 'geoms': names,
                                                'penetration_m': float(-c.dist)})
        path_collision_samples += int(sample_deep)
data.qpos[:] = final_qpos
mujoco.mj_forward(model, data)
joint_margins = []
for j in range(model.njnt):
    if model.jnt_limited[j] and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
        q = data.qpos[model.jnt_qposadr[j]]
        low, high = model.jnt_range[j]
        joint_margins.extend((float(q-low), float(high-q)))
report = {'scope': 'Position-only four-arm IK seed with fixed robot bases, no time parameterization or WBC execution',
          'roles': results, 'deep_nonfloor_robot_contacts': deep_contacts,
          'deep_nonfloor_robot_contact_count': len(deep_contacts),
          'worst_nonfloor_robot_penetration_m': max((max(0.,-x['distance_m']) for x in deep_contacts), default=0.),
          'staged_path_segments': len(path_keyframes)-1,
          'interpolated_path_collision_samples_of_404': path_collision_samples,
          'interpolated_path_worst_nonfloor_robot_penetration_m': path_deepest,
          'interpolated_path_worst_contacts': sorted(path_collision_examples,
                                                     key=lambda x: -x['penetration_m'])[:12],
          'minimum_final_joint_limit_margin_rad': min(joint_margins),
          'all_positions_within_1cm': all(r['final_position_error_m'] < .01 for r in results.values()),
          'collision_free_at_seed': not deep_contacts}
(OUT / 'reach_ik_probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
np.savez_compressed(OUT / 'reach_ik_seed.npz', qpos=data.qpos, initial_qpos=initial_qpos,
                    path_qpos=np.stack([q for _, q in path_keyframes]),
                    path_labels=np.array([name for name, _ in path_keyframes]))
print(json.dumps({'errors_m': {k: v['final_position_error_m'] for k, v in results.items()},
                  'deep_contacts': len(deep_contacts),
                  'all_positions_within_1cm': report['all_positions_within_1cm']}))
