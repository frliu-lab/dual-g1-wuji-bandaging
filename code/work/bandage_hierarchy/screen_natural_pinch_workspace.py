"""Find a collision-safe Wuji pinch aperture near the bandage free tab."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5.xml'))
d=mujoco.MjData(m)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
q0=d.qpos.copy();q0[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
screen=json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
goal_q=q0.copy();goal_q[[m.joint(n).qposadr[0] for n in screen['joint_names']]]=screen['results'][0]['joint_targets_rad']
names=[m.joint(j).name for j in range(m.njnt) if
       m.joint(j).name.startswith(('g1_b/right_wrist_',
                                     'g1_b/wuji_right/r_thumb_',
                                     'g1_b/wuji_right/r_index_finger_')) and
       'abd' not in m.joint(j).name]
jids=[m.joint(name).id for name in names]
qa=np.array([m.jnt_qposadr[j] for j in jids])
lo=m.jnt_range[jids,0]+1e-4;hi=m.jnt_range[jids,1]-1e-4
tipids=[m.body(n).id for n in ('g1_b/wuji_right/r_thumb_tip_sensor_frame',
                               'g1_b/wuji_right/r_index_finger_tip_sensor_frame')]
center=np.asarray(screen['edge_center_m'])
rng=np.random.default_rng(20261122)
rows=[]
for base_name,base in [('planted',q0),('reach',goal_q)]:
    start=base[qa].copy()
    for trial in range(16):
        x0=start if trial==0 else np.clip(start+rng.normal(0,.3,len(start)),lo,hi)
        def evaluate(x):
            d.qpos[:]=base;d.qpos[qa]=x;mujoco.mj_forward(m,d)
            tips=np.array([d.xpos[bid] for bid in tipids])
            pen=0.
            for c in d.contact:
                if c.dist>=0:continue
                geoms=[int(g) for g in c.geom if g>=0]
                bodies=[m.body(m.geom_bodyid[g]).name for g in geoms]
                if any(n.startswith('g1_b/') for n in bodies) and 'floor' not in [m.geom(g).name for g in geoms]:
                    pen=max(pen,-float(c.dist))
            return tips,pen
        def residual(x):
            tips,pen=evaluate(x)
            difference=tips[0]-tips[1]
            aperture=np.linalg.norm(difference)
            midpoint=tips.mean(axis=0)
            # Keep a gap for a physical cloth tab, rather than collapsing digits.
            return np.r_[.2*max(0.,aperture-.025),
                         .2*max(0.,.018-aperture),
                         .12*(midpoint-center),
                         20*max(0.,pen-.0025),
                         .0015*(x-start)]
        fit=least_squares(residual,x0,bounds=(lo,hi),max_nfev=70)
        tips,pen=evaluate(fit.x)
        aperture=float(np.linalg.norm(tips[0]-tips[1]))
        midpoint=tips.mean(axis=0)
        rows.append({'base':base_name,'trial':trial,'success':bool(fit.success),
                     'aperture_m':aperture,'midpoint_error_m':float(np.linalg.norm(midpoint-center)),
                     'max_b_nonfloor_penetration_m':pen,
                     'geometric_candidate':bool(.018<=aperture<=.025 and
                                                np.linalg.norm(midpoint-center)<.03 and pen<.003),
                     'tip_positions_m':tips.tolist(),'joint_targets_rad':fit.x.tolist()})
rows.sort(key=lambda r:(not r['geometric_candidate'],r['max_b_nonfloor_penetration_m'],r['midpoint_error_m'],r['aperture_m']))
report={'scope':'Static hand aperture only; no cloth opposing contacts, force closure, or dynamic grasp',
        'joint_names':names,'results':rows}
(OUT/'joint_b_natural_pinch_workspace_screen.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'candidates':sum(r['geometric_candidate'] for r in rows),
                  'best':{k:v for k,v in rows[0].items() if k!='joint_targets_rad'}}))
