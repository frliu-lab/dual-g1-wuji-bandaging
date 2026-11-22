"""Measure actual fingertip surface gap for hand-aperture candidates."""
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5.xml'))
d=mujoco.MjData(m)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
q0=d.qpos.copy();q0[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
reach=json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
goal=q0.copy();goal[[m.joint(n).qposadr[0] for n in reach['joint_names']]]=reach['results'][0]['joint_targets_rad']
screen=json.loads((OUT/'joint_b_natural_pinch_workspace_screen.json').read_text())
qa=np.array([m.joint(n).qposadr[0] for n in screen['joint_names']])
thumb=[g for g in range(m.ngeom) if m.geom_bodyid[g]==m.body('g1_b/wuji_right/r_thumb_tip_sensor_frame').id]
index=[g for g in range(m.ngeom) if m.geom_bodyid[g]==m.body('g1_b/wuji_right/r_index_finger_tip_sensor_frame').id]
rows=[]
for row in screen['results']:
    d.qpos[:]=q0 if row['base']=='planted' else goal
    d.qpos[qa]=row['joint_targets_rad']
    mujoco.mj_forward(m,d)
    pairs=[]
    for a in thumb:
        for b in index:
            endpoints=np.zeros(6)
            dist=float(mujoco.mj_geomDistance(m,d,a,b,.2,endpoints))
            pairs.append({'distance_m':dist,'thumb_geom':a,'index_geom':b,
                          'nearest_points_m':endpoints.reshape(2,3).tolist()})
    best=min(pairs,key=lambda x:x['distance_m'])
    rows.append({'base':row['base'],'trial':row['trial'],
                 'tip_center_aperture_m':row['aperture_m'],
                 'tip_surface_gap_m':best['distance_m'],
                 'midpoint_error_m':row['midpoint_error_m'],
                 'max_b_nonfloor_penetration_m':row['max_b_nonfloor_penetration_m'],
                 'nearest_pair':best})
rows.sort(key=lambda r:(abs(r['tip_surface_gap_m']-.008)+r['midpoint_error_m']+
                        10*max(0,r['max_b_nonfloor_penetration_m']-.003)))
report={'scope':'Static Wuji fingertip collision-surface gap, no cloth grasp or force',
        'candidates':rows}
(OUT/'joint_b_natural_pinch_surface_gap_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(rows[:6]))
