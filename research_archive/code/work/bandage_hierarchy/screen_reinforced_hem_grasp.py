"""Static opposed digit-load screen for a sewn bandage-end grasp hem."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
SCENE=OUT/'joint_dual_g1_pinchtab_extended20mm_reinforced_hem.xml'
PRIOR=OUT/'joint_b_right_loaded_pinch_static_refine_radius2mm.json'
prior=json.loads(PRIOR.read_text())
start=prior['top_passing_clearance_force'][0]
m=mujoco.MjModel.from_xml_path(str(SCENE))
d=mujoco.MjData(m)
q=d.qpos.copy();q[:167]=np.load(OUT/'joint_b_pinchtab_rrt_path.npz')['qpos'][-1,:167]
pad=np.load(OUT/'joint_b_flex_a_thumb_side_contact_candidates.npz')['qpos'][0]
for j in range(m.njnt):
    if m.joint(j).name.startswith(('g1_a/wuji_right/r_thumb_',
                                   'g1_a/wuji_right/r_index_finger_')):
        adr=int(m.jnt_qposadr[j]);q[adr]=pad[adr]
for name in ('g1_b/wuji_right/r_middle_finger_mcp_flex',
             'g1_b/wuji_right/r_middle_finger_pip',
             'g1_b/wuji_right/r_middle_finger_dip'):
    q[m.joint(name).qposadr[0]]+=.5
names=prior['joint_names']
jids=np.array([m.joint(name).id for name in names]);qa=m.jnt_qposadr[jids]
low,high=m.jnt_range[jids,0]+.01,m.jnt_range[jids,1]-.01
center=np.array(start['joint_targets_rad'])
rng=np.random.default_rng(20261127)
rows=[]
for trial in range(6001):
    sigma=0. if trial==0 else .025 if trial<=3000 else .05
    d.qpos[:]=q
    d.qpos[qa]=np.clip(center+rng.normal(0.,sigma,len(qa)),low,high)
    d.qvel[:]=0.;mujoco.mj_forward(m,d)
    forces={'thumb':0.,'index':0.};normals={'thumb':[],'index':[]};max_pen=0.
    for ci,c in enumerate(d.contact):
        geoms=[int(g) for g in c.geom if g>=0]
        bodies=[m.body(m.geom_bodyid[g]).name for g in geoms]
        if c.dist<0 and any(b.startswith('g1_b/') for b in bodies):
            if 'floor' not in [m.geom(g).name for g in geoms]:max_pen=max(max_pen,-float(c.dist))
        if not any(m.geom(g).name=='bandage_reinforced_hem_grasp' for g in geoms):continue
        for digit in forces:
            if any(b.startswith(f'g1_b/wuji_right/r_{digit}_') for b in bodies):
                wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                forces[digit]+=max(0.,float(wrench[0]))
                normals[digit].append(c.frame[:3].copy())
    opposed=any(float(a@b)<-.5 for a in normals['thumb'] for b in normals['index'])
    row={'trial':trial,'digit_normal_force_n':forces,'min_digit_normal_force_n':min(forces.values()),
         'opposed':opposed,'max_b_nonfloor_penetration_m':max_pen,
         'joint_targets_rad':d.qpos[qa].tolist()}
    row['static_screen_pass']=bool(opposed and row['min_digit_normal_force_n']>=.5 and max_pen<=.003)
    rows.append(row)
passing=[r for r in rows if r['static_screen_pass']]
safe=[r for r in rows if r['opposed'] and r['max_b_nonfloor_penetration_m']<=.003]
report={'scope':'Instantaneous MuJoCo static posture screen for a modeled sewn hem; no motion, no independent hold or hardware evidence',
        'scene':SCENE.name,'samples':len(rows),'joint_names':names,
        'static_pass_count':len(passing),'baseline':rows[0],
        'best_safe':max(safe,key=lambda r:r['min_digit_normal_force_n'],default=None),
        'top_passing':sorted(passing,key=lambda r:-r['min_digit_normal_force_n']+200*max(0.,r['max_b_nonfloor_penetration_m']-.002))[:30]}
(OUT/'joint_b_reinforced_hem_static_grasp_screen.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:None if report[k] is None else report[k]['min_digit_normal_force_n']
                  for k in ('baseline','best_safe')}|{'static_pass_count':len(passing)}))
