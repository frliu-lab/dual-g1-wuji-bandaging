"""Retract B-left unused middle finger away from tray while preserving the handle pinch."""
import itertools
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--yaw-deg',choices=(-30,20),type=int,default=-30)
args=parser.parse_args()
STEM = ('joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyaw'+
        ('m30p0' if args.yaw_deg==-30 else '20p0')+'_spoolz0p702_handle')
model = mujoco.MjModel.from_xml_path(str(OUT / f'{STEM}_variant.xml'))
data = mujoco.MjData(model)
original = np.load(OUT / f'{STEM}_candidate.npz')['qpos'].copy()
names = [f'g1_b/wuji_left/l_middle_finger_{x}' for x in ('mcp_flex','mcp_abd','pip','dip')]
qa = np.array([model.joint(name).qposadr[0] for name in names])
rows=[]
for mcp,abd,pip,dip in itertools.product((0.,.3,.6,.9),(0.,-.2,.2),(.0,.4,.8,1.2),(.0,.4,.8)):
    q=original.copy();q[qa]=[mcp,abd,pip,dip]
    data.qpos[:]=q;mujoco.mj_forward(model,data)
    middle_force=0.;deep=0;digit_force={'thumb':0.,'index':0.}
    digit_normal={'thumb':np.zeros(3),'index':np.zeros(3)}
    for ci,c in enumerate(data.contact):
        if c.geom1<0 or c.geom2<0:continue
        geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
        bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                model.body(model.geom_bodyid[c.geom2]).name)
        if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
            deep+=1
        wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
        force=max(0.,float(wrench[0]))
        if any(b.startswith('g1_b/wuji_left/l_middle_finger_') for b in bodies):middle_force+=force
        if 'bandage_handle_prototype' in geoms:
            for digit in digit_force:
                if any(b.startswith(f'g1_b/wuji_left/l_{digit}_') for b in bodies):
                    digit_force[digit]+=force
                    sign=1. if geoms[0]=='bandage_handle_prototype' else -1.
                    digit_normal[digit]+=sign*force*np.asarray(c.frame[:3])
    a,b=digit_normal.values();denom=np.linalg.norm(a)*np.linalg.norm(b)
    cosine=float(a@b/denom) if denom>1e-8 else None
    feasible=bool(deep==0 and middle_force<.05 and all(.05<f<=2. for f in digit_force.values())
                  and cosine is not None and cosine<-.5)
    rows.append({'middle_joints_rad':[mcp,abd,pip,dip], 'middle_nonfloor_force_n':middle_force,
                 'deep_contact_count':deep,'handle_digit_forces_n':digit_force,
                 'handle_normal_cosine':cosine,'static_contact_gate':feasible})
passing=[r for r in rows if r['static_contact_gate']]
passing.sort(key=lambda r:sum(abs(x) for x in r['middle_joints_rad']))
if passing:
    best=passing[0]
    q=original.copy();q[qa]=best['middle_joints_rad']
    np.savez_compressed(OUT / f'{STEM}_middle_retracted_candidate.npz',qpos=q)
else:best=None
report={'scope':'Static B-left unused middle finger retraction; no WBC, trajectory, or bandaging claim',
        'candidate_count':len(rows),'passing_count':len(passing),'best':best,
        'all_candidates':rows}
(OUT/('joint_spool_handle_middle_retraction_summary.json' if args.yaw_deg==-30 else
      'joint_spool_handle_middle_retraction_yaw20_summary.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'candidate_count':len(rows),'passing_count':len(passing),'best':best}))
