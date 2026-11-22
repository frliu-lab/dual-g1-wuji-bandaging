"""Measure one-cycle Wuji fingertip force authority with full MuJoCo rollouts."""
import json
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC

OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--horizon-s',type=float,default=.02)
args=parser.parse_args()
if args.horizon_s<=0 or args.horizon_s>.2:parser.error('--horizon-s must be in (0, 0.2]')
m=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'))
d=mujoco.MjData(m)
d.qpos[:167]=np.load(OUT/'joint_b_pinchtab_rrt_path.npz')['qpos'][-1,:167]
mujoco.mj_forward(m,d)
tip_names=('g1_b/wuji_right/r_thumb_tip_sensor_frame','g1_b/wuji_right/r_index_finger_tip_sensor_frame')
palms=('g1_a/wuji_left/l_wrist','g1_a/wuji_right/r_wrist','g1_b/wuji_left/l_wrist','g1_b/wuji_right/r_wrist')
def relevant_contact(model,c):
    return any(f>=0 for f in c.flex) or any(model.body(model.geom_bodyid[g]).name.startswith(('g1_a/','g1_b/')) for g in c.geom if g>=0)
wbc=JointWBC(m,d.qpos.copy(),palm_names=palms,contact_filter=relevant_contact)
wbc.backend='DIRECT';wbc.flex_tip_normal_goal_n=1.;wbc.hand_position_mode=True
for i in wbc.hand_indices:wbc.hand_commands[i]=d.qpos[wbc.act_qadr[i]]
pinch=[int(i) for i in wbc.hand_indices if m.actuator(int(i)).name.startswith('g1_b/wuji_right/r_') and any(t in m.actuator(int(i)).name for t in ('THJ','FFJ'))]
wbc.force_control_indices=set(pinch)
ctrl,_,_=wbc.solve(d)
initial_edge=d.flexvert_xpos[-2:].mean(axis=0).copy()

def trial(adjustments=None):
    live=mujoco.MjData(m)
    live.qpos[:]=d.qpos;live.qvel[:]=d.qvel;live.act[:]=d.act
    live.eq_active[:]=d.eq_active;live.time=d.time;live.ctrl[:]=ctrl
    for actuator,delta in (adjustments or {}).items():
        live.ctrl[actuator]=np.clip(live.ctrl[actuator]+delta,*m.actuator_ctrlrange[actuator])
    mujoco.mj_forward(m,live)
    total=np.zeros(2);max_pen=0.;n=round(args.horizon_s/m.opt.timestep)
    for _ in range(n):
        mujoco.mj_step(m,live)
        for ci,c in enumerate(live.contact):
            if not any(f>=0 for f in c.flex):continue
            bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
            for digit,name in enumerate(tip_names):
                if name in bodies:
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,live,ci,wrench)
                    total[digit]+=max(0.,float(wrench[0]))
            if c.dist<0 and any(name.startswith('g1_b/') for name in bodies):max_pen=max(max_pen,-float(c.dist))
    return {'mean_tip_normal_force_n':(total/n).tolist(),
            'edge_displacement_m':float(np.linalg.norm(live.flexvert_xpos[-2:].mean(axis=0)-initial_edge)),
            'max_b_flex_penetration_m':max_pen}

baseline=trial()
cases=[]
for i in pinch:
    for delta in (-.1,-.05,.05,.1):
        result=trial({i:delta})
        cases.append({'actuator':m.actuator(i).name,'delta_ctrl_rad':delta,**result})
def actuator(token):return next(i for i in pinch if m.actuator(i).name.endswith(token))
combinations=[]
for values in ({'THJ1':.1,'FFJ3':.1},{'THJ1':.1,'THJ0':.1},
               {'THJ1':.05,'THJ0':.1,'FFJ3':.1},{'THJ1':.1,'FFJ1':.1,'FFJ3':.1},
               {'THJ1':.05,'THJ0':.05,'THJ3':-.05,'FFJ3':.05}):
    result=trial({actuator(name):delta for name,delta in values.items()})
    combinations.append({'delta_ctrl_rad':values,**result})
report={'scope':'One constant-control MuJoCo counterfactual from the same initial state; fixture held and prewrapped bandage; not closed-loop grasp',
        'horizon_s':args.horizon_s,
        'baseline':baseline,'cases':cases,'combinations':combinations}
(OUT/f'joint_b_pinchtab_pinch_control_authority_{round(args.horizon_s*1000)}ms.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps({'baseline':baseline,'best_thumb':max(cases,key=lambda x:x['mean_tip_normal_force_n'][0]),
                  'best_index':max(cases,key=lambda x:x['mean_tip_normal_force_n'][1]),
                  'combinations':combinations}))

