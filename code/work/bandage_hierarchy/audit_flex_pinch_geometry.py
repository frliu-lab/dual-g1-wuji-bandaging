"""Check whether static free-tab reach actually has opposed fingertip contacts."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
scene=OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5.xml'
m=mujoco.MjModel.from_xml_path(str(scene))
d=mujoco.MjData(m)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
screen=json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
q0=d.qpos.copy();q0[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
qa=[m.joint(name).qposadr[0] for name in screen['joint_names']]
goal=np.asarray(screen['results'][0]['joint_targets_rad'])
cases=[]
for label,q in [('start',q0),('goal',q0.copy())]:
    if label=='goal':q[qa]=goal
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    digit_contacts={digit:[] for digit in ('thumb','index')}
    all_robot=[]
    for c in d.contact:
        if not any(f>=0 for f in c.flex):continue
        geoms=[int(g) for g in c.geom if g>=0]
        bodies=[m.body(m.geom_bodyid[g]).name for g in geoms]
        robot=next((name for name in bodies if name.startswith('g1_b/wuji_right/')),None)
        if robot is None:continue
        entry={'body':robot,'distance_m':float(c.dist),
               'contact_position_m':list(map(float,c.pos)),
               'normal_world':list(map(float,c.frame[:3]))}
        all_robot.append(entry)
        if 'thumb_tip_sensor_frame' in robot:digit_contacts['thumb'].append(entry)
        if 'index_finger_tip_sensor_frame' in robot:digit_contacts['index'].append(entry)
    pairs=[]
    for a in digit_contacts['thumb']:
        for b in digit_contacts['index']:
            dot=float(np.dot(a['normal_world'],b['normal_world']))
            pairs.append({'normal_dot':dot,'tip_contact_separation_m':float(np.linalg.norm(
                np.asarray(a['contact_position_m'])-np.asarray(b['contact_position_m']))),
                'max_distance_m':max(a['distance_m'],b['distance_m'])})
    cases.append({'pose':label,'thumb_tip_flex_contacts':len(digit_contacts['thumb']),
                  'index_tip_flex_contacts':len(digit_contacts['index']),
                  'opposing_pair_count_dot_lt_minus_0p5':sum(p['normal_dot']<-.5 and p['max_distance_m']<.003 for p in pairs),
                  'best_opposing_pair':min(pairs,key=lambda x:x['normal_dot']) if pairs else None,
                  'all_b_right_flex_contacts':all_robot})
report={'scope':'Static contact geometry only; no normal force or sustained grasp claim','cases':cases}
(OUT/'joint_b_flex_pinch_contact_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{k:v for k,v in c.items() if k!='all_b_right_flex_contacts'} for c in cases]))
