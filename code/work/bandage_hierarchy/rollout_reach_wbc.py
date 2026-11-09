"""Closed-loop four-arm reach to the precontact seed; no object grasp or bandaging."""
import argparse
import json
import sys
import subprocess
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
sys.path.insert(0, str(ROOT / 'work/wuji_multiagent'))
from joint_wbc import JointWBC
from controller_router import HandExpertRouter, Observation, Mode
from scene_evidence import hand_contacts

parser = argparse.ArgumentParser()
parser.add_argument('--hz', type=int, default=50)
parser.add_argument('--segment-s', type=float, default=1.0)
parser.add_argument('--hold-s', type=float, default=1.0)
parser.add_argument('--video', action='store_true')
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
for robot in ('g1_a', 'g1_b'):
    feet = [i for i in range(model.ngeom) if model.geom(i).name.startswith(robot+'/')
            and 'official_foot_contact' in model.geom(i).name]
    clearance = min(float(data.geom_xpos[i, 2]-model.geom_size[i, 0]) for i in feet)
    root = model.joint(f'{robot}/floating_base').id
    data.qpos[model.jnt_qposadr[root]+2] -= clearance+.0002
mujoco.mj_forward(model, data)
seed = np.load(OUT / 'reach_ik_seed.npz')
path = seed['path_qpos'].copy()
if np.max(np.abs(path[0]-data.qpos)) > 1e-6:
    raise RuntimeError('IK seed and current scene initialization differ')

palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
def robot_contact_only(m, c):
    return any(m.body(m.geom_bodyid[int(g)]).name.startswith(('g1_a/', 'g1_b/'))
               for g in c.geom if g >= 0)

wbc = JointWBC(model, data.qpos.copy(), palm_names=palms,
               contact_filter=robot_contact_only)
finger_qadr = np.array([int(model.jnt_qposadr[model.actuator_trnid[i, 0]])
                       for i in wbc.hand_indices], dtype=int)
router = HandExpertRouter(len(finger_qadr))
contact_observations = np.zeros(4, dtype=int)
router_modes = {mode.value: 0 for mode in Mode}
renderer = None
video_frames = []
if args.video:
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 540
    renderer = mujoco.Renderer(model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .80]
    camera.distance = 2.8
    camera.azimuth = 90
    camera.elevation = -18
    visual = mujoco.MjvOption()
    visual.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
    visual.geomgroup[2] = 0
    next_frame_s = 0.
interval = round((1/args.hz)/model.opt.timestep)
if abs(interval*model.opt.timestep-1/args.hz) > 1e-9:
    raise ValueError('WBC period must be an integer number of simulation steps')
per_segment = round(args.segment_s*args.hz)
feet = [i for i in range(model.ngeom) if 'official_foot_contact' in model.geom(i).name]
root_ids = [model.body(f'{robot}/pelvis').id for robot in ('g1_a', 'g1_b')]
root_start = data.xpos[root_ids].copy()
max_pelvis_shift = 0.
max_joint_error = 0.
max_penetration = 0.
deep_contact_samples = 0
both_foot_support = 0
steps = 0
error = None
worst_contact = None
arm_qadr = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
            if any(model.joint(j).name.startswith(f'{r}/{s}_') for r in ('g1_a','g1_b') for s in ('left','right'))
            and any(t in model.joint(j).name for t in ('shoulder','elbow','wrist'))
            and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]

schedule = [(segment, k, per_segment) for segment in range(1, len(path))
            for k in range(per_segment)]
schedule += [(len(path)-1, per_segment, per_segment)]*round(args.hold_s*args.hz)
for segment, k, count in schedule:
        u = (k+1)/count
        ease = u*u*(3-2*u)
        wbc.qref[:] = (1-ease)*path[segment-1]+ease*path[segment]
        try:
            evidence = hand_contacts(model, data)
            contact_observations += np.asarray(evidence, dtype=int)
            decision = router.step(Observation(
                time_s=float(data.time), phase='approach', body_valid=True,
                feet_supported=both_foot_support == steps if steps else True,
                arm_pose_stable=False, hand_contacts=evidence,
                hand_feedback_fresh=False), data.qpos[finger_qadr].copy(), None)
            router_modes[decision.mode.value] += 1
            ctrl, _, _ = wbc.solve(data)
            data.ctrl[:] = ctrl
            for _ in range(interval):
                mujoco.mj_step(model, data)
                steps += 1
                if renderer is not None and data.time >= next_frame_s:
                    renderer.update_scene(data, camera=camera, scene_option=visual)
                    video_frames.append(renderer.render().copy())
                    next_frame_s += 1/15
                max_pelvis_shift = max(max_pelvis_shift, float(np.max(np.linalg.norm(
                    data.xpos[root_ids]-root_start, axis=1))))
                supported = set()
                for c in data.contact:
                    if c.geom1 < 0 or c.geom2 < 0:
                        continue
                    names = (model.geom(c.geom1).name, model.geom(c.geom2).name)
                    bodies = (model.body(model.geom_bodyid[c.geom1]).name,
                              model.body(model.geom_bodyid[c.geom2]).name)
                    if 'floor' in names:
                        for b in bodies:
                            if b.startswith('g1_a/'): supported.add('g1_a')
                            if b.startswith('g1_b/'): supported.add('g1_b')
                    elif any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
                        if c.dist < -.003:
                            deep_contact_samples += 1
                            if -c.dist > max_penetration:
                                max_penetration = float(-c.dist)
                                worst_contact = {'bodies': bodies, 'distance_m': float(c.dist),
                                                 'time_s': float(data.time)}
                both_foot_support += int(supported == {'g1_a','g1_b'})
            max_joint_error = max(max_joint_error, float(np.max(np.abs(
                data.qpos[arm_qadr]-wbc.qref[arm_qadr]))))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            break
report = {'scope': 'Four-arm precontact WBC reach only; no grasp, cloth or bandaging',
          'wbc_hz': args.hz, 'segment_s': args.segment_s, 'hold_s': args.hold_s,
          'segments_completed': segment if not error else segment-1,
          'simulated_time_s': float(data.time),
          'both_robot_foot_support_fraction': both_foot_support/max(steps,1),
          'max_pelvis_shift_m': max_pelvis_shift,
          'max_arm_joint_tracking_error_rad': max_joint_error,
          'deep_nonfloor_robot_contact_samples': deep_contact_samples,
          'worst_nonfloor_robot_penetration_m': max_penetration,
          'worst_contact': worst_contact,
          'real_contact_observations_by_role': contact_observations.tolist(),
          'router_observer_mode_counts': router_modes,
          'router_output_actuated': False,
          'final_arm_joint_error_rad': float(np.max(np.abs(data.qpos[arm_qadr]-path[-1,arm_qadr]))),
          'worst_final_arm_joints': sorted(({'joint': model.joint(model.actuator_trnid[i,0]).name,
                                             'error_rad': float(abs(data.qpos[q]-path[-1,q]))}
                                            for q in arm_qadr for i in range(model.nu)
                                            if model.jnt_qposadr[model.actuator_trnid[i,0]] == q),
                                           key=lambda item: -item['error_rad'])[:8],
          'final_fingertip_position_error_m': {
              name: float(np.linalg.norm(data.xpos[model.body(name).id]-goal))
              for name, goal in (
                  ('g1_a/wuji_left/l_index_finger_tip_sensor_frame', (-.22,.18,.84)),
                  ('g1_a/wuji_right/r_index_finger_tip_sensor_frame', (-.38,-.44,.72)),
                  ('g1_b/wuji_left/l_index_finger_tip_sensor_frame', (.38,-.44,.78)),
                  ('g1_b/wuji_right/r_index_finger_tip_sensor_frame', (.45,-.08,.89)))},
          'finite_state': bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
          'error': error}
if renderer is not None:
    renderer.close()
    ffmpeg = ROOT / 'work/retarget_env/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    video = OUT / f'four_arm_reach_{args.hz}hz_{args.segment_s:g}s.mp4'
    proc = subprocess.Popen([str(ffmpeg), '-y', '-loglevel', 'error', '-f', 'rawvideo',
                             '-pixel_format', 'rgb24', '-video_size', '960x540', '-framerate', '15',
                             '-i', 'pipe:0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)],
                            stdin=subprocess.PIPE)
    for frame in video_frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    report['video_frames'] = len(video_frames)
(OUT / f'four_arm_reach_{args.hz}hz_{args.segment_s:g}s.json').write_text(
    json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
