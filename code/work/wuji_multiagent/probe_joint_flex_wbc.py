"""One-frame dual-G1 WBC feasibility probe with flexible strip contacts."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np

from joint_wbc import JointWBC

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--radius-mm',type=float,default=52.)
parser.add_argument('--phase-deg',type=float,default=0.)
parser.add_argument('--all-contacts',action='store_true')
parser.add_argument('--flex-and-robot',action='store_true')
args=parser.parse_args()
variant=(f'_radius{args.radius_mm:g}mm_phase{args.phase_deg:g}deg'
         if args.radius_mm!=52. or args.phase_deg else '')
model=mujoco.MjModel.from_xml_path(str(OUT/f'joint_dual_g1_prewrapped_flex_scene_world_anchored{variant}.xml'))
data=mujoco.MjData(model)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
q=np.asarray(seed['samples'][-1]['qpos'])
data.qpos[:len(q)]=q
mujoco.mj_forward(model,data)
palms=('g1_a/wuji_left/l_wrist','g1_a/wuji_right/r_wrist',
       'g1_b/wuji_left/l_wrist','g1_b/wuji_right/r_wrist')
def robot_contact_only(m,c):
    return any(m.body(m.geom_bodyid[g]).name.startswith(('g1_a/','g1_b/'))
               for g in c.geom if g>=0)
def flex_and_robot(m,c):
    return any(f>=0 for f in c.flex) or robot_contact_only(m,c)
wbc=JointWBC(model,data.qpos.copy(),palm_names=palms,
             contact_filter=None if args.all_contacts else flex_and_robot if args.flex_and_robot else robot_contact_only)
wbc.backend='DIRECT'
error=None;meta=None
worst=[]
try:
    _,meta,_=wbc.solve(data)
    from direct_wbc_qp import solve_direct
    acceleration,_,_,_=solve_direct(wbc.last_numeric)
    numeric=wbc.last_numeric
    residual=numeric['J']@acceleration-numeric['target']
    for i,contact_id in enumerate(wbc.contact_indices):
        local=residual[3*i:3*i+3]
        finite=local[np.isfinite(numeric['target'][3*i:3*i+3])]
        c=data.contact[contact_id]
        rigid_geom=next((g for g in c.geom if g>=0),None)
        worst.append({'max_contact_acceleration_residual_m_s2':float(np.max(np.abs(finite))) if len(finite) else 0.,
                      'is_flex':bool(any(f>=0 for f in c.flex)),
                      'rigid_body':model.body(model.geom_bodyid[rigid_geom]).name if rigid_geom is not None else None,
                      'penetration_m':max(0.,-float(c.dist))})
    worst=sorted(worst,key=lambda x:x['max_contact_acceleration_residual_m_s2'],reverse=True)[:10]
except Exception as exc:
    error=f'{type(exc).__name__}: {exc}'
report={'scope':'One-frame dual-G1 WBC with prewrapped flex contacts; no rollout or wrapping',
        'radius_mm':args.radius_mm,'phase_deg':args.phase_deg,
        'all_contacts':args.all_contacts,
        'flex_and_robot':args.flex_and_robot,
        'wbc_error':error,'wbc_meta':meta,
        'initial_flex_contact_count':sum(any(f>=0 for f in c.flex) for c in data.contact),
        'contact_rows_in_wbc':len(wbc.contact_indices),
        'worst_contact_task_residuals':worst,
        'finite_state':bool(np.all(np.isfinite(data.qpos)))}
(OUT/f'joint_flex_wbc_one_frame_probe{variant}{"_all_contacts" if args.all_contacts else "_flex_and_robot" if args.flex_and_robot else ""}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
