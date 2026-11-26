"""Finite-difference B-left middle-finger clearance at the measured worst contact."""
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'))
d=mujoco.MjData(m)
state=np.load(OUT/'joint_b_pinchtab_worst_b_penetration_state.npz')
base=state['qpos'];d.qpos[:]=base
js=[j for j in range(m.njnt) if m.joint(j).name.startswith('g1_b/wuji_left/l_middle_finger_')]
def contact_distance():
    mujoco.mj_forward(m,d)
    return min((float(c.dist) for c in d.contact
                if c.geom1>=0 and c.geom2>=0 and
                'bandage_handle_prototype' in (m.geom(c.geom1).name,m.geom(c.geom2).name) and
                any(m.body(m.geom_bodyid[g]).name.startswith('g1_b/wuji_left/l_middle_finger_') for g in c.geom)),default=.01)
baseline=contact_distance();cases=[]
for j in js:
    adr=int(m.jnt_qposadr[j]);q=float(base[adr]);lo,hi=m.jnt_range[j]
    for delta in (-.1,-.05,-.02,.02,.05,.1):
        if q+delta<lo+.02 or q+delta>hi-.02:continue
        d.qpos[adr]=q+delta
        cases.append({'joint':m.joint(j).name,'delta_rad':delta,'handle_clearance_m':contact_distance()})
        d.qpos[adr]=q
report={'scope':'Static single-joint perturbation at the saved worst dynamic contact; no trajectory safety proof',
        'time_s':float(state['time_s']),'baseline_clearance_m':baseline,'cases':cases}
(OUT/'joint_b_left_middle_handle_clearance_probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'baseline_m':baseline,'best':sorted(cases,key=lambda x:x['handle_clearance_m'],reverse=True)[:8]}))
