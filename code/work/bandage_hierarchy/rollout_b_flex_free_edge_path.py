"""Closed-loop dual-G1 WBC test of the collision-screened free-edge approach."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC

OUT = ROOT/'outputs/wuji_multiagent/bandage_v0'
SCENE = OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5.xml'
PATH = OUT/'joint_b_flex_free_edge_rrt_path.npz'
m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)
path_file = np.load(PATH)
waypoints = path_file['qpos']
joint_names = [str(s) for s in path_file['joint_names']]
qa = np.array([m.joint(s).qposadr[0] for s in joint_names])
va = np.array([m.joint(s).dofadr[0] for s in joint_names])
d.qpos[:] = waypoints[0]
mujoco.mj_forward(m, d)

palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
def relevant_contact(model, c):
    return any(f >= 0 for f in c.flex) or any(
        model.body(model.geom_bodyid[g]).name.startswith(('g1_a/', 'g1_b/'))
        for g in c.geom if g >= 0)

wbc = JointWBC(m, d.qpos.copy(), palm_names=palms, contact_filter=relevant_contact)
wbc.backend = 'DIRECT'
interval = round(.02/m.opt.timestep)
duration = 1.0
initial_edge = d.flexvert_xpos[64:66].mean(axis=0).copy()
max_pen = 0.0
max_tip_error = 0.0
feet_count = 0
flex_contact_count = 0
edge_drift = 0.0
solves = 0
error = None
trace = []

def ref_at(u):
    scaled = np.clip(u, 0, 1)*(len(waypoints)-1)
    i = min(len(waypoints)-2, int(scaled))
    t = scaled-i
    return waypoints[i]*(1-t)+waypoints[i+1]*t

try:
    for k in range(round(duration/.02)):
        progress = (k+1)/round(duration/.02)
        smooth = progress*progress*(3-2*progress)
        wbc.qref[:] = ref_at(smooth)
        ctrl, meta, _ = wbc.solve(d)
        d.ctrl[:] = ctrl
        solves += 1
        for _ in range(interval):
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                raise RuntimeError('Nonfinite state')
            feet = {'a': set(), 'b': set()}
            has_forearm_flex = False
            for c in d.contact:
                geoms = [int(g) for g in c.geom if g >= 0]
                names = [m.geom(g).name for g in geoms]
                bodies = [m.body(m.geom_bodyid[g]).name for g in geoms]
                flex = any(f >= 0 for f in c.flex)
                if flex and 'training_forearm_dynamic' in bodies:
                    has_forearm_flex = True
                if c.dist < 0 and 'floor' not in names and any(n.startswith('g1_b/') for n in bodies):
                    max_pen = max(max_pen, -float(c.dist))
                if 'floor' in names:
                    for agent in ('a', 'b'):
                        for side in ('left', 'right'):
                            if any(n.startswith(f'g1_{agent}/{side}_official_foot_contact_') for n in names):
                                feet[agent].add(side)
            feet_count += all(feet[a] == {'left', 'right'} for a in ('a', 'b'))
            flex_contact_count += has_forearm_flex
            edge_drift = max(edge_drift, float(np.linalg.norm(d.flexvert_xpos[64:66].mean(axis=0)-initial_edge)))
        tip_error = float(np.max(np.abs(d.qpos[qa]-wbc.qref[qa])))
        max_tip_error = max(max_tip_error, tip_error)
        trace.append({'time_s': float(d.time), 'path_progress': float(smooth),
                      'max_joint_tracking_error_rad': tip_error,
                      'free_edge_drift_m': float(np.linalg.norm(d.flexvert_xpos[64:66].mean(axis=0)-initial_edge)),
                      'contact_task_residual_m_s2': meta['contact_acceleration_task_residual_m_s2']})
except Exception as exc:
    error = f'{type(exc).__name__}: {exc}'

steps = max(1, solves*interval)
report = {'scope': 'Dynamic WBC tracking of a static path toward a flexible free tab; no force-closed grasp or completed wrap',
          'duration_s': float(d.time), 'planned_duration_s': duration, 'control_solves': solves,
          'error': error, 'two_robot_two_foot_fraction': feet_count/steps,
          'flex_forearm_contact_fraction': flex_contact_count/steps,
          'max_b_nonfloor_penetration_m': max_pen,
          'max_free_edge_drift_m': edge_drift,
          'max_joint_tracking_error_rad': max_tip_error,
          'final_joint_target_error_rad': float(np.max(np.abs(d.qpos[qa]-waypoints[-1, qa]))),
          'trace': trace}
(OUT/'joint_b_flex_free_edge_dynamic_path_rollout.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
