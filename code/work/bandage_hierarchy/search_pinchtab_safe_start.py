"""Find a collision-safe B-right pregrasp for the revised cloth tab."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab_bulged.xml'))
d=mujoco.MjData(m)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
base=d.qpos.copy();base[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
reach=json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
names=reach['joint_names'];jids=[m.joint(n).id for n in names]
qa=np.array([m.jnt_qposadr[j] for j in jids])
lo=m.jnt_range[jids,0]+1e-4;hi=m.jnt_range[jids,1]-1e-4
start=base[qa].copy();wrist_id=m.body('g1_b/wuji_right/r_wrist').id
d.qpos[:]=base;mujoco.mj_forward(m,d);wrist0=d.xpos[wrist_id].copy()
rng=np.random.default_rng(20261122);rows=[]
for trial in range(10):
    x0=start if trial==0 else np.clip(start+rng.normal(0,.15,len(start)),lo,hi)
    def evaluate(x):
        d.qpos[:]=base;d.qpos[qa]=x;mujoco.mj_forward(m,d)
        pen=0.
        for c in d.contact:
            if c.dist>=0:continue
            geoms=[int(g) for g in c.geom if g>=0]
            bodies=[m.body(m.geom_bodyid[g]).name for g in geoms]
            if any(n.startswith('g1_b/') for n in bodies) and 'floor' not in [m.geom(g).name for g in geoms]:
                pen=max(pen,-float(c.dist))
        return pen,d.xpos[wrist_id].copy()
    def residual(x):
        pen,wrist=evaluate(x)
        return np.r_[20*max(0.,pen-.002),.05*(wrist-wrist0),.001*(x-start)]
    fit=least_squares(residual,x0,bounds=(lo,hi),max_nfev=80)
    pen,wrist=evaluate(fit.x)
    rows.append({'trial':trial,'optimizer_success':bool(fit.success),
                 'max_b_nonfloor_penetration_m':pen,
                 'wrist_displacement_m':float(np.linalg.norm(wrist-wrist0)),
                 'safe_start':bool(pen<.003),
                 'joint_targets_rad':fit.x.tolist()})
rows.sort(key=lambda r:(not r['safe_start'],r['wrist_displacement_m'],r['max_b_nonfloor_penetration_m']))
report={'scope':'Static revised-tab pregrasp with other scene bodies fixed; no dynamic entry path',
        'joint_names':names,'results':rows}
(OUT/'joint_b_pinchtab_safe_start_screen.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passing':sum(r['safe_start'] for r in rows),
                  'best':{k:v for k,v in rows[0].items() if k!='joint_targets_rad'}}))
